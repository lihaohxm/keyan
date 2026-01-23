function assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights)
%QOE_AWARE Greedy QoE-aware assignment.

if nargin < 6
    weights = cfg.weights(1, :);
end

num_users = cfg.num_users;
num_ris = cfg.num_ris;

p_watts = 10.^(p_dbw / 10);
p_k = p_watts / num_users;

cost = zeros(num_users, num_ris + 1);

for k = 1:num_users
    for l = 0:num_ris
        if l == 0
            h_eff = ch.h_d(:, k);
        else
            theta = ch.theta(:, l);
            h_eff = ch.h_d(:, k) + ch.G(:, :, l) * (theta .* ch.H_ris(:, k, l));
        end
        gain = norm(h_eff).^2;
        gamma = p_k * gain / (cfg.noise_watts + cfg.eps);
        xi = semantic_map(gamma, cfg.m_k, semantic_mode, table_path, struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
        [avg_qoe, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, weights);
        cost(k, l + 1) = avg_qoe;
    end
end

capacity = cfg.k0 * ones(num_ris, 1);
assign = zeros(num_users, 1);

[~, order] = sort(min(cost, [], 2));

for idx = 1:num_users
    k = order(idx);
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
