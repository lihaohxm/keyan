function [avg_qoe, qoe_vec, meta] = qoe(cfg, gamma, M, xi, weights, prop_delay)
%QOE Compute QoE cost for each user.
%
% 改进的QoE公式：对SINR连续敏感，不仅仅是二值惩罚
%
% Qd = w_d * (T_tx / d_k)           - 归一化时延（连续）
% Qs = w_s * D                       - 语义失真（连续）
% 惩罚 = 违约时的额外惩罚（较小）

num_users = numel(gamma);

if nargin < 5 || isempty(weights)
    weights = cfg.weights(1, :);
end
if nargin < 6 || isempty(prop_delay)
    prop_delay = zeros(num_users, 1);
end

w_d = weights(1);
w_s = weights(2);

rho = cfg.rho;

% 计算时延
T_tx_only = rho .* M ./ (cfg.bandwidth * log2(1 + gamma(:) + cfg.eps) + cfg.eps);
T_tx = T_tx_only + prop_delay(:);

% 计算语义失真
D = 1 - xi(:);

% 获取deadline
num_hard = round(cfg.hard_ratio * num_users);
hard_mask = false(num_users, 1);
hard_mask(1:num_hard) = true;

if numel(cfg.deadlines) == 2
    d_vec = cfg.deadlines(1) * ones(num_users, 1);
    d_vec(~hard_mask) = cfg.deadlines(2);
else
    d_vec = cfg.deadlines(1) * ones(num_users, 1);
end

% ========== 改进的QoE公式 ==========
% 基础部分：归一化值（对SINR连续敏感）
% 时延：T_tx / d_k，值越小越好
% 语义：D，值越小越好

Qd_base = T_tx ./ d_vec;  % 归一化时延，若=1表示刚好满足deadline
Qs_base = D ./ cfg.dmax;  % 归一化失真，若=1表示刚好满足阈值

% 违约惩罚（较小，只作为额外惩罚）
penalty_d = (T_tx > d_vec) .* cfg.b_d .* ((T_tx - d_vec) ./ d_vec);  % 比例惩罚
penalty_s = (D > cfg.dmax) .* cfg.b_s .* ((D - cfg.dmax) ./ cfg.dmax);

% 总QoE（越小越好）
Qd = Qd_base + penalty_d;
Qs = Qs_base + penalty_s;

qoe_vec = w_d .* Qd + w_s .* Qs;
avg_qoe = mean(qoe_vec);

% 元数据
meta.deadline_ok = mean(T_tx <= d_vec);
meta.semantic_ok = mean(D <= cfg.dmax);
meta.delay_violation_rate = mean(T_tx > d_vec);
meta.semantic_violation_rate = mean(D > cfg.dmax);
meta.T_tx = T_tx;
meta.D = D;
meta.Qd = Qd;
meta.Qs = Qs;
end
