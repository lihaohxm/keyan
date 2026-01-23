function assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights)
%QOE_AWARE Greedy QoE-aware assignment (minimize per-user QoE cost).
%
% Key fixes vs previous version:
%   1) Order "hard" users first: users with larger best-possible cost go earlier.
%   2) Use the same effective channel model as evaluation (includes cfg.ris_gain).
%   3) Use SINR computed with full multi-user interference (matched-filter beamforming),
%      consistent with sinr_rate.m.

if nargin < 6
    weights = cfg.weights(1, :);
end

num_users = cfg.num_users;
num_ris = cfg.num_ris;

% Precompute per-user per-choice cost: QoE cost of user k when ONLY user k uses RIS l
% (others direct-link). This is a stable and reasonably fair surrogate.
cost = zeros(num_users, num_ris + 1);

% baseline assignment: nobody uses RIS
base_assign = zeros(num_users, 1);

for k = 1:num_users
    for l = 0:num_ris
        tmp_assign = base_assign;
        tmp_assign(k) = l;

        h_eff = effective_channel(cfg, ch, tmp_assign);
        gamma = sinr_rate(cfg, h_eff, p_dbw); % returns gamma as first output

        % semantic mapping (vector form)
        xi = semantic_map(gamma, cfg.m_k, semantic_mode, table_path, ...
            struct('a', cfg.proxy_a, 'b', cfg.proxy_b));

        [~, qoe_vec, ~] = qoe(cfg, gamma, cfg.m_k, xi, weights);

        cost(k, l + 1) = qoe_vec(k); % per-user cost
    end
end

% Capacity per RIS (each RIS can serve up to k0 users)
capacity = cfg.k0 * ones(num_ris, 1);
assign = zeros(num_users, 1);

% ---- FIX: assign "hard" users first (larger best-possible cost) ----
best_cost = min(cost, [], 2);
[~, order] = sort(best_cost, 'descend');

for idx = 1:num_users
    k = order(idx);

    % choices: pick lowest cost first
    [~, choices] = sort(cost(k, :), 'ascend');

    assigned = false;
    for c = 1:numel(choices)
        l = choices(c) - 1;

        if l == 0
            assign(k) = 0;
            assigned = true;
            break;
        elseif capacity(l) > 0
            assign(k) = l;
            capacity(l) = capacity(l) - 1;
            assigned = true;
            break;
        end
    end

    if ~assigned
        assign(k) = 0;
    end
end
end
