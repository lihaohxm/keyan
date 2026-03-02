function [V, c, rates, sum_rate, gamma_c, gamma_p] = rsma_wmmse(cfg, h_eff, p_dbw, weights, max_iter)
%RSMA_WMMSE 严格单层 RSMA 波束成形更新 (基于 QoE 权重)
%
% 实现论文 3.2 节算法：
%   - 变量: v_c (公共预编码), v_k (私有预编码), c_k (公共速率分配)
%   - 目标: 最大化 sum(w_k * (R_k_private + c_k))
%
% 输入:
%   h_eff   - 等效信道 (nt x num_users)
%   p_dbw   - 发射功率 (dBW)
%   weights - QoE 权重链式求导得到的权重 (num_users x 1)
%
% 输出:
%   V        - {v_c, v_k} 预编码集合
%   c        - 公共速率分配向量
%   sum_rate - 系统总速率 (包含公共+私有)

num_users = size(h_eff, 2);
nt = cfg.nt;
p_watts = 10.^(p_dbw / 10);
noise_power = cfg.noise_watts;

if nargin < 4 || isempty(weights)
    weights = ones(num_users, 1);
end

% Ensure column vector and normalize scale for numerical stability.
if isrow(weights)
    weights = weights';
end
m_w = mean(weights);
if ~isfinite(m_w) || m_w <= cfg.eps
    weights = ones(num_users, 1);
else
    weights = weights / (m_w + cfg.eps);
end
if nargin < 5
    max_iter = 10;
end

% 1. 初始化波束 (MRT)
v_c = sum(h_eff, 2);
v_c = v_c / (norm(v_c) + cfg.eps) * sqrt(p_watts * 0.2); % 20% 功率给公共流
V_p = h_eff;
for k = 1:num_users
    V_p(:, k) = V_p(:, k) / (norm(V_p(:, k)) + cfg.eps) * sqrt(p_watts * 0.8 / num_users);
end

% 初始化接收器、权重、速率分配
u_c = zeros(num_users, 1);
u_p = zeros(num_users, 1);
w_c = zeros(num_users, 1);
w_p = zeros(num_users, 1);
c = (1/num_users) * ones(num_users, 1);

denom_c_vec = zeros(num_users, 1);
denom_p_vec = zeros(num_users, 1);

% 迭代更新
for iter = 1:max_iter
    % --- Step A: 更新 MMSE 接收器 u ---
    for k = 1:num_users
        hk = h_eff(:, k);
        % 公共流
        denom_c = abs(hk' * v_c)^2 + sum(abs(hk' * V_p).^2) + noise_power;
        denom_c_vec(k) = denom_c;
        u_c(k) = (hk' * v_c) / (denom_c + cfg.eps);
        % 私有流
        denom_p = sum(abs(hk' * V_p).^2) + noise_power;
        denom_p_vec(k) = denom_p;
        u_p(k) = (hk' * V_p(:, k)) / (denom_p + cfg.eps);
    end

    % --- Step B: 更新 MSE 权重 w ---
    e_c = zeros(num_users, 1);
    e_p = zeros(num_users, 1);
    for k = 1:num_users
        hk = h_eff(:, k);
        sc = abs(hk' * v_c)^2;
        sp = abs(hk' * V_p(:, k))^2;

        % With MMSE receivers, MSE admits a closed-form: e = 1 - S / (S + I + N)
        e_c(k) = 1 - sc / (denom_c_vec(k) + cfg.eps);
        e_p(k) = 1 - sp / (denom_p_vec(k) + cfg.eps);

        % Numerical safety
        e_c(k) = max(real(e_c(k)), 1e-12);
        e_p(k) = max(real(e_p(k)), 1e-12);

        w_c(k) = 1 / e_c(k);
        w_p(k) = 1 / e_p(k);
    end

    % --- Step C: 更新预编码 V (闭式解/二次规划) ---
    % 构造公共流预编码相关矩阵 A_common
    A_common = zeros(nt, nt);
    for i = 1:num_users
        hi = h_eff(:, i);
        A_common = A_common + weights(i) * w_c(i) * abs(u_c(i))^2 * (hi * hi');
    end
    % 构造私有流更新矩阵 A_all (对所有 v_k 相同)
    A_all = A_common;
    for i = 1:num_users
        hi = h_eff(:, i);
        A_all = A_all + weights(i) * w_p(i) * abs(u_p(i))^2 * (hi * hi');
    end

    % RHS vectors
    b_c = zeros(nt, 1);
    for i = 1:num_users
        b_c = b_c + weights(i) * w_c(i) * conj(u_c(i)) * h_eff(:, i);
    end
    b_scale = (weights(:) .* w_p(:) .* conj(u_p(:))).'; % 1 x K
    B_p = h_eff .* (ones(nt, 1) * b_scale); % nt x K

    % Lagrange multiplier mu via bisection (power-tight & stable)
    mu_low = 0;
    mu_high = 1e-8;

    [v_c_tmp, V_p_tmp, p_tmp] = update_precoders(A_common, A_all, b_c, B_p, mu_low, cfg.eps);
    if p_tmp <= p_watts
        v_c = v_c_tmp;
        V_p = V_p_tmp;
    else
        % Increase upper bound until feasible
        for t = 1:60
            [~, ~, p_hi] = update_precoders(A_common, A_all, b_c, B_p, mu_high, cfg.eps);
            if p_hi <= p_watts
                break;
            end
            mu_high = mu_high * 2;
        end

        % Bisection
        for t = 1:30
            mu_mid = 0.5 * (mu_low + mu_high);
            [v_c_mid, V_p_mid, p_mid] = update_precoders(A_common, A_all, b_c, B_p, mu_mid, cfg.eps);
            if p_mid > p_watts
                mu_low = mu_mid;
            else
                mu_high = mu_mid;
                v_c = v_c_mid;
                V_p = V_p_mid;
            end
        end
    end
end

% 计算最终速率
gamma_c = zeros(num_users, 1);
gamma_p = zeros(num_users, 1);
for k = 1:num_users
    hk = h_eff(:, k);
    gamma_c(k) = abs(hk' * v_c)^2 / (sum(abs(hk' * V_p).^2) + noise_power + cfg.eps);
    gamma_p(k) = abs(hk' * V_p(:, k))^2 / (sum(abs(hk' * V_p(:, setdiff(1:num_users, k))).^2) + noise_power + cfg.eps);
end

R_c_limit = cfg.bandwidth * log2(1 + min(gamma_c) + cfg.eps);
R_p = cfg.bandwidth * log2(1 + gamma_p + cfg.eps);

% 公共速率分配 c (按权重分配)
total_weight = sum(weights);
c = (weights(:) / (total_weight + cfg.eps)) * R_c_limit;

rates = R_p + c;
sum_rate = sum(rates);

V.v_c = v_c;
V.V_p = V_p;
V.c = c;
end

function [v_c, V_p, p_used] = update_precoders(A_common, A_all, b_c, B_p, mu, eps0)
nt = size(A_common, 1);
I = eye(nt);
v_c = (A_common + mu * I + eps0 * I) \ b_c;
V_p = (A_all + mu * I + eps0 * I) \ B_p;
p_used = real(norm(v_c)^2 + sum(sum(abs(V_p).^2)));
end