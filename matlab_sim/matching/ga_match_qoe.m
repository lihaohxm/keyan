function [bestChoice, bestQoE, histBest] = ga_match_qoe(cfg, ch, p_dbw, opts)
%GA_MATCH_QOE Genetic Algorithm for RIS-user matching
%
% IMPORTANT: This GA should optimize QoE Cost (not Sum-Rate) to serve as
% a fair benchmark against the proposed algorithm.
%
% Key features:
%   - Strong elitism (top 5 never modified)
%   - Random injection (3 fresh individuals per generation)
%   - Flip mutation (0->RIS or RIS->0)
%   - Mutation rate = 0.2

if nargin < 4, opts = struct(); end

% GA Parameters
Np     = get_opt(opts, 'Np', 20);    % Population size
Niter  = get_opt(opts, 'Niter', 5);  % Iterations
Pc     = get_opt(opts, 'Pc', 0.80);
Pm     = get_opt(opts, 'Pm', 0.20);
elite  = get_opt(opts, 'elite', 3);
inject = get_opt(opts, 'inject', 2);
t      = get_opt(opts, 't', 3);
verbose = get_opt(opts, 'verbose', false);
semantic_mode = get_opt(opts, 'semantic_mode', cfg.semantic_mode);
table_path = get_opt(opts, 'table_path', cfg.semantic_table);
weights = get_opt(opts, 'weights', cfg.weights(1,:));
geom = get_opt(opts, 'geom', []);  % Geometry for propagation delay

K = cfg.num_users;
L = cfg.num_ris;
K0 = cfg.k0;

% All users can use RIS (no mask restriction when hard_ratio=1.0)
num_hard = round(cfg.hard_ratio * K);
can_use_ris = false(K, 1);
can_use_ris(1:num_hard) = true;
ris_users = find(can_use_ris);
direct_only = find(~can_use_ris);

% ========== INITIALIZATION ==========
% Option: use proposed solution as seed (elitism)
use_proposed_seed = get_opt(opts, 'use_proposed_seed', false);

pop = zeros(K, Np);
for i = 1:Np
    pop(:, i) = random_assignment(K, L, K0, ris_users, direct_only);
end

% If use_proposed_seed is enabled, replace first individual with proposed solution
if use_proposed_seed
    try
        proposed_assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights, geom);
        proposed_assign(direct_only) = 0;
        proposed_assign = repair(proposed_assign, L, K0);
        pop(:, 1) = proposed_assign;
    catch
        % If qoe_aware fails, keep random initialization
    end
end

% ========== FITNESS ==========
% Default: optimize QoE Cost (minimize), not Sum-Rate
% This ensures GA is a fair benchmark against proposed algorithm
optimize_sumrate = get_opt(opts, 'optimize_sumrate', false);  % Default FALSE!

    function fit_val = evaluate(choice)
        if ~is_feasible(choice, L, K0)
            fit_val = -1e9;
            return;
        end
        h_eff = effective_channel(cfg, ch, choice);
        if optimize_sumrate
            % Legacy mode: maximize sum-rate (not recommended for QoE experiments)
            [~, ~, fit_val] = sinr_rate(cfg, h_eff, p_dbw);
        else
            % QoE optimization mode: minimize QoE Cost (with propagation delay)
            [gamma, ~, ~] = sinr_rate(cfg, h_eff, p_dbw);
            % Use same proxy as proposed
            proxy_params = struct('a', cfg.proxy_a, 'b', cfg.proxy_b);
            xi = semantic_map(gamma, cfg.m_k, semantic_mode, table_path, proxy_params);
            prop_delay = calc_prop_delay_ga(choice, geom, cfg);
            [qoe_cost, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, weights, prop_delay);
            fit_val = -qoe_cost;  % Negative because GA maximizes fitness
        end
    end

% Evaluate initial population
fit = zeros(1, Np);
for i = 1:Np
    fit(i) = evaluate(pop(:, i));
end

[bestFit, bestIdx] = max(fit);
bestChoice = pop(:, bestIdx);
histBest = zeros(1, Niter + 1);
histBest(1) = bestFit;

% ========== EVOLUTION ==========
for gen = 1:Niter
    
    % Elitism
    [sorted_fit, sortIdx] = sort(fit, 'descend');
    elitePop = pop(:, sortIdx(1:elite));
    eliteFit = sorted_fit(1:elite);
    
    % Tournament selection
    mating_pool = zeros(K, Np);
    for i = 1:Np
        cands = randi(Np, 1, t);
        [~, w] = max(fit(cands));
        mating_pool(:, i) = pop(:, cands(w));
    end
    
    % Crossover & Mutation
    n_offspring = Np - elite - inject;
    offspring = zeros(K, n_offspring);
    
    for i = 1:2:n_offspring
        p1 = mating_pool(:, randi(Np));
        p2 = mating_pool(:, randi(Np));
        
        % Single-point crossover
        if rand < Pc
            pt = randi(K);
            c1 = [p1(1:pt); p2(pt+1:end)];
            c2 = [p2(1:pt); p1(pt+1:end)];
        else
            c1 = p1; c2 = p2;
        end
        
        % FLIP MUTATION (0<->RIS)
        c1 = flip_mutate(c1, L, ris_users, Pm);
        c2 = flip_mutate(c2, L, ris_users, Pm);
        
        % Enforce constraints
        c1(direct_only) = 0;
        c2(direct_only) = 0;
        c1 = repair(c1, L, K0);
        c2 = repair(c2, L, K0);
        
        offspring(:, i) = c1;
        if i + 1 <= n_offspring
            offspring(:, i + 1) = c2;
        end
    end
    
    % Random injection
    injected = zeros(K, inject);
    for i = 1:inject
        injected(:, i) = random_assignment(K, L, K0, ris_users, direct_only);
    end
    
    % Combine
    pop = [elitePop, offspring, injected];
    fit(1:elite) = eliteFit;
    for i = elite+1:Np
        fit(i) = evaluate(pop(:, i));
    end
    
    % Update best
    [gen_best, gen_best_idx] = max(fit);
    if gen_best > bestFit
        bestFit = gen_best;
        bestChoice = pop(:, gen_best_idx);
    end
    histBest(gen + 1) = bestFit;
end

% Output QoE
h_eff = effective_channel(cfg, ch, bestChoice);
[gamma, ~, ~] = sinr_rate(cfg, h_eff, p_dbw);
proxy_params = struct('mode', 'sigmoid', 'c', 0.6, 'gamma0', 0);
xi = semantic_map(gamma, cfg.m_k, semantic_mode, table_path, proxy_params);
prop_delay = calc_prop_delay_ga(bestChoice, geom, cfg);
[bestQoE, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, weights, prop_delay);

if verbose
    fprintf('GA: fit=%.4f, qoe=%.4f, ris_users=%d\n', bestFit, bestQoE, sum(bestChoice>0));
end
end

% ========== HELPERS ==========

function v = get_opt(opts, f, d)
    if isfield(opts, f), v = opts.(f); else, v = d; end
end

function choice = random_assignment(K, L, K0, ris_users, direct_only)
    choice = zeros(K, 1);
    cap = K0 * ones(L, 1);
    order = randperm(numel(ris_users));
    for i = 1:numel(order)
        k = ris_users(order(i));
        avail = find(cap > 0);
        opts = [0; avail(:)];
        pick = opts(randi(numel(opts)));
        choice(k) = pick;
        if pick > 0, cap(pick) = cap(pick) - 1; end
    end
    choice(direct_only) = 0;
end

function ok = is_feasible(choice, L, K0)
    ok = true;
    for l = 1:L
        if sum(choice == l) > K0, ok = false; return; end
    end
end

function choice = repair(choice, L, K0)
    for l = 1:L
        users = find(choice == l);
        excess = numel(users) - K0;
        if excess > 0
            victims = users(randperm(numel(users), excess));
            choice(victims) = 0;
        end
    end
end

function child = flip_mutate(child, L, ris_users, Pm)
    % FLIP MUTATION: randomly flip 0<->RIS for diversity
    for i = 1:numel(ris_users)
        k = ris_users(i);
        if rand < Pm
            if child(k) == 0
                child(k) = randi([1, L]);   % direct -> random RIS
            else
                if rand < 0.5
                    child(k) = 0;           % RIS -> direct
                else
                    child(k) = randi([1, L]); % RIS -> different RIS
                end
            end
        end
    end
end

function prop_delay = calc_prop_delay_ga(assign, geom, cfg)
    % Calculate propagation delay for QoE evaluation in GA
    num_users = numel(assign);
    prop_delay = zeros(num_users, 1);
    prop_delay_factor = 1e-5;  % Must match qoe_aware.m
    
    if isempty(geom) || ~isfield(geom, 'ris') || ~isfield(geom, 'ue')
        return;
    end
    
    bs_pos = geom.bs;
    if numel(bs_pos) == 3, bs_pos = bs_pos(1:2); end
    
    for k = 1:num_users
        l = assign(k);
        if l > 0 && l <= size(geom.ris, 1)
            ris_pos = geom.ris(l, :);
            ue_pos = geom.ue(k, :);
            if numel(ris_pos) == 3, ris_pos = ris_pos(1:2); end
            if numel(ue_pos) == 3, ue_pos = ue_pos(1:2); end
            
            d_bs_ris = norm(ris_pos - bs_pos);
            d_ris_ue = norm(ue_pos - ris_pos);
            total_dist = d_bs_ris + d_ris_ue;
            prop_delay(k) = prop_delay_factor * total_dist;
        end
    end
end
