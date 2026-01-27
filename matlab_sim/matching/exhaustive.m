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
    [gamma, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
    
    if optimize_qoe
        % Compute QoE cost WITH propagation delay (consistent with evaluation)
        xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
            struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
        prop_delay = calc_prop_delay_internal(candidate, geom, cfg);
        [qoe_cost, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, weights, prop_delay);
        
        if qoe_cost < best_cost
            best_cost = qoe_cost;
            assign = candidate;
        end
    else
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

function prop_delay = calc_prop_delay_internal(assign, geom, cfg)
    % Calculate propagation delay for QoE optimization
    num_users = numel(assign);
    prop_delay = zeros(num_users, 1);
    prop_delay_factor = 1e-5;  % Must match qoe_aware.m
    
    if isempty(geom) || ~isfield(geom, 'ris') || ~isfield(geom, 'ue')
        return;
    end
    
    bs_pos = geom.bs(1, :);
    
    for k = 1:num_users
        l = assign(k);
        if l > 0 && l <= size(geom.ris, 1)
            ris_pos = geom.ris(l, :);
            ue_pos = geom.ue(k, :);
            d_bs_ris = norm(ris_pos - bs_pos);
            d_ris_ue = norm(ue_pos - ris_pos);
            total_dist = d_bs_ris + d_ris_ue;
            prop_delay(k) = prop_delay_factor * total_dist;
        end
    end
end
