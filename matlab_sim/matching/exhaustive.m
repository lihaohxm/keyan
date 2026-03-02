function [assign, info] = exhaustive(cfg, ch, p_dbw, opts)
%EXHAUSTIVE Exhaustive assignment for small sizes.
%
% opts.optimize_qoe = true: minimize QoE cost (default: false, maximize sum-rate)
% opts.geom = geometry struct for propagation delay calculation (optional)

if nargin < 4, opts = struct(); end

info.skipped = false;
info.reason = '';

% (L+1)^K combinations - allow up to 500000 (5^8 = 390625)
max_combos = 500000;
num_combos = (cfg.num_ris + 1)^cfg.num_users;

if num_combos > max_combos
    info.skipped = true;
    info.reason = sprintf('Too many combos (%d > %d)', num_combos, max_combos);
    warning('exhaustive: %s. Falling back to norm_based.', info.reason);
    assign = norm_based(cfg, ch);
    return;
end

% Check optimization mode
optimize_qoe = false;
if isfield(opts, 'optimize_qoe'), optimize_qoe = opts.optimize_qoe; end

% Get geometry for propagation delay
geom = [];
if isfield(opts, 'geom'), geom = opts.geom; end

% Get weights (default from cfg)
weights = cfg.weights(1,:);
if isfield(opts, 'weights'), weights = opts.weights; end

num_users = cfg.num_users;
num_ris = cfg.num_ris;

if optimize_qoe
    best_cost = inf;  % Minimize QoE cost
else
    best_cost = -inf; % Maximize sum-rate
end
assign = zeros(num_users, 1);

options = 0:num_ris;

combos = cell(1, num_users);
for k = 1:num_users
    combos{k} = options;
end

idxs = cell(1, num_users);
[idxs{:}] = ndgrid(combos{:});

all_assign = zeros(numel(idxs{1}), num_users);
for k = 1:num_users
    all_assign(:, k) = idxs{k}(:);
end

for row = 1:size(all_assign, 1)
    candidate = all_assign(row, :).';
    if ~capacity_ok(candidate, cfg)
        continue;
    end
    h_eff = effective_channel(cfg, ch, candidate);
    
    if optimize_qoe
        [V_eval, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(num_users, 1), 3);
        profile = make_uniform_profile(cfg, num_users, weights);
        sol = struct('assign', candidate, 'theta_all', ch.theta, 'V', V_eval);
        if isfield(cfg, 'proxy_a') && isfield(cfg, 'proxy_b')
            sem_params = struct('a', cfg.proxy_a, 'b', cfg.proxy_b);
        else
            sem_params = struct();
        end
        eval_opts = struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table, ...
            'semantic_params', sem_params);
        out = evaluate_system_rsma(cfg, ch, geom, sol, profile, eval_opts);
        qoe_cost = out.avg_qoe;
        
        if qoe_cost < best_cost
            best_cost = qoe_cost;
            assign = candidate;
        end
    else
        [~, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
        if sum_rate > best_cost
            best_cost = sum_rate;
            assign = candidate;
        end
    end
end
end

function ok = capacity_ok(assign, cfg)
    ok = true;
    for l = 1:cfg.num_ris
        if sum(assign == l) > cfg.k0
            ok = false;
            return;
        end
    end
end

function profile = make_uniform_profile(cfg, K, weights)
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


