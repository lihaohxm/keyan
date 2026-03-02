function [bestChoice, bestQoE, histBest, ga_info] = ga_match_qoe(cfg, ch, p_dbw, opts)
%GA_MATCH_QOE Genetic Algorithm for RIS-user matching.
%
% Fairness notes:
% 1) theta strategy is consistent in fitness and final evaluation.
% 2) GA fitness can directly use external per-user profile (urgent/normal).
%
% theta_strategy:
%   'random_fixed' (default): use ch.theta consistently
%   'align_fixed'           : use per-assignment aligned theta consistently

if nargin < 4, opts = struct(); end

% GA Parameters
Np = get_opt(opts, 'pop_size', get_opt(opts, 'Np', 20));
Niter = get_opt(opts, 'num_generations', get_opt(opts, 'Niter', 5));
Pc = get_opt(opts, 'Pc', 0.80);
Pm = get_opt(opts, 'Pm', 0.20);
elite = get_opt(opts, 'elite', 3);
inject = get_opt(opts, 'inject', 2);
t = get_opt(opts, 't', 3);
verbose = get_opt(opts, 'verbose', false);
enable_ga_log = get_opt(opts, 'ga_log', false) || verbose;
budget_evals = get_opt(opts, 'budget_evals', inf); % GA-budgeted support

semantic_mode = get_opt(opts, 'semantic_mode', cfg.semantic_mode);
table_path = get_opt(opts, 'table_path', cfg.semantic_table);
geom = get_opt(opts, 'geom', []);
theta_strategy = lower(char(get_opt(opts, 'theta_strategy', 'random_fixed')));

if isfield(cfg, 'proxy_a') && isfield(cfg, 'proxy_b')
    proxy_params = struct('a', cfg.proxy_a, 'b', cfg.proxy_b);
else
    proxy_params = struct();
end

K = cfg.num_users;
L = cfg.num_ris;
K0 = cfg.k0;

profile_opt = get_opt(opts, 'profile', []);
weights_opt = get_opt(opts, 'weights', cfg.weights(1, :));
if ~isempty(profile_opt)
    profile_eval = profile_opt;
else
    % Legacy fallback (kept for compatibility with older callers).
    profile_eval = make_uniform_profile(cfg, weights_opt);
end

num_hard = round(cfg.hard_ratio * K);
can_use_ris = false(K, 1);
can_use_ris(1:num_hard) = true;
ris_users = find(can_use_ris);
direct_only = find(~can_use_ris);

if isfinite(budget_evals) && budget_evals > 0
    % Keep the same mutation/crossover style but shrink GA scale to budget.
    Np = min(Np, max(4, floor(budget_evals / 2)));
end
elite = min(max(1, elite), max(1, Np - 2));
inject = min(max(0, inject), max(0, Np - elite - 1));
n_offspring = max(1, Np - elite - inject);

if isfinite(budget_evals) && budget_evals > 0
    eval_per_gen = max(1, Np - elite);
    Niter = min(Niter, max(0, floor((budget_evals - Np) / eval_per_gen)));
end

theta_base = ch.theta;
eval_opts = struct('semantic_mode', semantic_mode, 'table_path', table_path, 'semantic_params', proxy_params);
eval_count = 0;

% ========== INITIALIZATION ==========
use_proposed_seed = get_opt(opts, 'use_proposed_seed', false);
seed_assignment = get_opt(opts, 'seed_assignment', []);
pop = zeros(K, Np);
for i = 1:Np
    pop(:, i) = random_assignment(K, L, K0, ris_users, direct_only);
end

if ~isempty(seed_assignment) && numel(seed_assignment) == K
    seed_assignment = seed_assignment(:);
    seed_assignment(direct_only) = 0;
    seed_assignment = repair(seed_assignment, L, K0);
    pop(:, 1) = seed_assignment;
end

if use_proposed_seed
    try
        proposed_assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, profile_eval.weights, geom, profile_eval);
        proposed_assign(direct_only) = 0;
        proposed_assign = repair(proposed_assign, L, K0);
        pop(:, 1) = proposed_assign;
    catch
    end
end

% ========== FITNESS ==========
fit = -1e9 * ones(1, Np);
for i = 1:Np
    [fit(i), ~] = evaluate(pop(:, i));
end

[bestFit, bestIdx] = max(fit);
bestChoice = pop(:, bestIdx);
histBest = zeros(1, Niter + 1);
histBest(1) = bestFit;

% ========== EVOLUTION ==========
for gen = 1:Niter
    if eval_count >= budget_evals
        break;
    end

    [sorted_fit, sortIdx] = sort(fit, 'descend');
    elitePop = pop(:, sortIdx(1:elite));
    eliteFit = sorted_fit(1:elite);

    mating_pool = zeros(K, Np);
    for i = 1:Np
        cands = randi(Np, 1, t);
        [~, w] = max(fit(cands));
        mating_pool(:, i) = pop(:, cands(w));
    end

    offspring = zeros(K, n_offspring);
    i = 1;
    while i <= n_offspring
        p1 = mating_pool(:, randi(Np));
        p2 = mating_pool(:, randi(Np));

        if rand < Pc
            pt = randi(K);
            c1 = [p1(1:pt); p2(pt + 1:end)];
            c2 = [p2(1:pt); p1(pt + 1:end)];
        else
            c1 = p1; c2 = p2;
        end

        c1 = flip_mutate(c1, L, ris_users, Pm);
        c2 = flip_mutate(c2, L, ris_users, Pm);
        c1(direct_only) = 0;
        c2(direct_only) = 0;
        c1 = repair(c1, L, K0);
        c2 = repair(c2, L, K0);

        offspring(:, i) = c1;
        if i + 1 <= n_offspring
            offspring(:, i + 1) = c2;
        end
        i = i + 2;
    end

    injected = zeros(K, inject);
    for i = 1:inject
        injected(:, i) = random_assignment(K, L, K0, ris_users, direct_only);
    end

    pop = [elitePop, offspring, injected];
    fit(1:elite) = eliteFit;
    for i = elite + 1:Np
        [fit(i), ~] = evaluate(pop(:, i));
    end

    [gen_best, gen_best_idx] = max(fit);
    if gen_best > bestFit
        bestFit = gen_best;
        bestChoice = pop(:, gen_best_idx);
    end
    histBest(gen + 1) = bestFit;

    if enable_ga_log
        [~, out_best] = evaluate_raw(bestChoice);
        if ~isempty(out_best)
            fprintf('[GA][gen %d] fit=%.6f qoe=%.6f delay_vio=%.4f sem_vio=%.4f eval=%d\n', ...
                gen, bestFit, out_best.avg_qoe, out_best.delay_vio_rate_all, out_best.semantic_vio_rate_all, eval_count);
        end
    end
end

[~, out_final] = evaluate_raw(bestChoice);
if isempty(out_final)
    bestQoE = inf;
else
    bestQoE = out_final.avg_qoe;
end

if enable_ga_log && ~isempty(out_final)
    fprintf('[GA][final] fit=%.6f qoe=%.6f delay_vio=%.4f sem_vio=%.4f eval=%d budget=%g\n', ...
        bestFit, out_final.avg_qoe, out_final.delay_vio_rate_all, out_final.semantic_vio_rate_all, eval_count, budget_evals);
end

ga_info = struct();
ga_info.eval_count = eval_count;
ga_info.theta_strategy = theta_strategy;
ga_info.budget_evals = budget_evals;
ga_info.Np = Np;
ga_info.Niter = Niter;
ga_info.pop_size = Np;
ga_info.num_generations = Niter;
ga_info.best_fit = bestFit;
ga_info.mode = ternary(isfinite(budget_evals), 'ga_rt', 'ga_ub');

    function [fit_val, out] = evaluate(choice)
        if eval_count >= budget_evals
            fit_val = -1e9;
            out = [];
            return;
        end
        [fit_val, out] = evaluate_raw(choice);
        eval_count = eval_count + 1;
    end

    function [fit_val, out] = evaluate_raw(choice)
        if ~is_feasible(choice, L, K0)
            fit_val = -1e9;
            out = [];
            return;
        end
        theta_eval = resolve_theta(theta_strategy, cfg, ch, choice, theta_base);
        h_eff = effective_channel(cfg, ch, choice, theta_eval);
        [V_eval, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(K, 1), 3);
        sol = struct('assign', choice, 'theta_all', theta_eval, 'V', V_eval);
        out = evaluate_system_rsma(cfg, ch, geom, sol, profile_eval, eval_opts);
        fit_val = -out.avg_qoe;
    end
end

function theta_eval = resolve_theta(theta_strategy, cfg, ch, choice, theta_base)
switch lower(theta_strategy)
    case 'align_fixed'
        theta_eval = build_align_theta(cfg, ch, choice, theta_base);
    otherwise
        theta_eval = theta_base;
end
end

function theta_all = build_align_theta(cfg, ch, assign, theta_base)
theta_all = theta_base;
L = cfg.num_ris;
for l = 1:L
    users_l = find(assign == l);
    if isempty(users_l)
        continue;
    end
    k = users_l(1);
    hdk = ch.h_d(:, k);
    w0 = hdk / (norm(hdk) + cfg.eps);
    proj = (w0' * ch.G(:, :, l)).' .* ch.H_ris(:, k, l);
    theta_all(:, l) = exp(-1j * angle(proj + cfg.eps));
end
end

function v = get_opt(opts, f, d)
if isfield(opts, f) && ~isempty(opts.(f))
    v = opts.(f);
else
    v = d;
end
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
    if pick > 0
        cap(pick) = cap(pick) - 1;
    end
end
choice(direct_only) = 0;
end

function ok = is_feasible(choice, L, K0)
ok = true;
for l = 1:L
    if sum(choice == l) > K0
        ok = false;
        return;
    end
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
for i = 1:numel(ris_users)
    k = ris_users(i);
    if rand < Pm
        if child(k) == 0
            child(k) = randi([1, L]);
        else
            if rand < 0.5
                child(k) = 0;
            else
                child(k) = randi([1, L]);
            end
        end
    end
end
end

function profile = make_uniform_profile(cfg, weights)
K = cfg.num_users;
profile = struct();
profile.M_k = cfg.m_k * ones(K, 1);
profile.weights = repmat(weights(:).', K, 1);
profile.d_k = default_deadlines(cfg, K);
profile.dmax_k = cfg.dmax * ones(K, 1);
end

function d_vec = default_deadlines(cfg, K)
num_hard = round(cfg.hard_ratio * K);
hard_mask = false(K, 1);
hard_mask(1:num_hard) = true;
if numel(cfg.deadlines) == 2
    d_vec = cfg.deadlines(1) * ones(K, 1);
    d_vec(~hard_mask) = cfg.deadlines(2);
else
    d_vec = cfg.deadlines(1) * ones(K, 1);
end
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
