function [avg_qoe, qoe_vec, meta] = qoe(cfg, gamma, M, xi, weights)
%QOE Compute QoE cost for each user.

num_users = numel(gamma);

if nargin < 5 || isempty(weights)
    weights = cfg.weights(1, :);
end

w_d = weights(1);
w_s = weights(2);

rho = cfg.rho;

T_tx = rho .* M ./ (cfg.bandwidth * log2(1 + gamma + cfg.eps) + cfg.eps);

D = 1 - xi;

num_hard = round(cfg.hard_ratio * num_users);

hard_mask = false(num_users, 1);
hard_mask(1:num_hard) = true;

hard_mask = hard_mask(randperm(num_users));

if numel(cfg.deadlines) == 2
    d_vec = cfg.deadlines(1) * ones(num_users, 1);
    d_vec(~hard_mask) = cfg.deadlines(2);
else
    d_vec = cfg.deadlines(1) * ones(num_users, 1);
end

sigmoid = @(x) 1 ./ (1 + exp(-x));

Qd = sigmoid((T_tx - d_vec) ./ cfg.beta_d) + (T_tx > d_vec) .* cfg.h_d .* cfg.b_d;
Qs = sigmoid((D - cfg.dmax) ./ cfg.beta_s) + (D > cfg.dmax) .* cfg.h_s .* cfg.b_s;

qoe_vec = w_d .* Qd + w_s .* Qs;
avg_qoe = mean(qoe_vec);

meta.deadline_ok = mean(T_tx <= d_vec);
meta.semantic_ok = mean(D <= cfg.dmax);
end
