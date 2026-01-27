function assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights, geom)
%QOE_AWARE QoE-aware greedy assignment algorithm.
%
% 核心思想：每一步选择能最大程度降低总QoE代价的(用户,RIS)配对
%
% 与其他算法的区别：
%   - Random: 随机分配
%   - Norm: 只看信道强度，不考虑QoE
%   - GA: 随机搜索，需要多次迭代
%   - Proposed: 基于QoE代价的最优贪婪，一次遍历

if nargin < 6 || isempty(weights)
    weights = cfg.weights(1, :);
end
if nargin < 7
    geom = [];
end

num_users = cfg.num_users;
num_ris = cfg.num_ris;

p_watts = 10.^(p_dbw / 10);
p_k = p_watts / num_users;

ris_gain = cfg.ris_gain;
prop_delay_factor = 1e-5;

w_d = weights(1);
w_s = weights(2);
rho = cfg.rho;
M = cfg.m_k;

% 获取每个用户的deadline
num_hard = round(cfg.hard_ratio * num_users);
hard_mask = false(num_users, 1);
hard_mask(1:num_hard) = true;

if numel(cfg.deadlines) == 2
    d_vec = cfg.deadlines(1) * ones(num_users, 1);
    d_vec(~hard_mask) = cfg.deadlines(2);
else
    d_vec = cfg.deadlines(1) * ones(num_users, 1);
end

% ==================== 构建代价矩阵 ====================
cost = zeros(num_users, num_ris + 1);  % 列1=直连, 列2~L+1=RIS

for k = 1:num_users
    d_k = d_vec(k);
    
    for l = 0:num_ris
        if l == 0
            h_eff = ch.h_d(:, k);
            prop_delay = 0;
        else
            theta = ch.theta(:, l);
            h_eff = ch.h_d(:, k) + ris_gain * ch.G(:, :, l) * (theta .* ch.H_ris(:, k, l));
            
            if ~isempty(geom) && isfield(geom, 'ris') && isfield(geom, 'ue')
                ris_pos = geom.ris(l, :);
                ue_pos = geom.ue(k, :);
                bs_pos = geom.bs(1, :);
                d_bs_ris = norm(ris_pos - bs_pos);
                d_ris_ue = norm(ue_pos - ris_pos);
                prop_delay = prop_delay_factor * (d_bs_ris + d_ris_ue);
            else
                prop_delay = 0;
            end
        end
        
        g = norm(h_eff).^2;
        gamma_val = p_k * g / (cfg.noise_watts + cfg.eps);
        
        T_tx_only = rho * M / (cfg.bandwidth * log2(1 + gamma_val + cfg.eps) + cfg.eps);
        T_tx = T_tx_only + prop_delay;
        
        xi = semantic_map(gamma_val, M, semantic_mode, table_path, struct());
        D = 1 - xi;
        
        % 与qoe.m一致的代价函数
        Qd_base = T_tx / d_k;
        Qs_base = D / cfg.dmax;
        penalty_d = (T_tx > d_k) * cfg.b_d * ((T_tx - d_k) / d_k);
        penalty_s = (D > cfg.dmax) * cfg.b_s * ((D - cfg.dmax) / cfg.dmax);
        Qd = Qd_base + penalty_d;
        Qs = Qs_base + penalty_s;
        
        qoe_cost = w_d * Qd + w_s * Qs;
        cost(k, l + 1) = qoe_cost;
    end
end

% ==================== 贪婪分配：每步选最优配对 ====================
capacity = cfg.k0 * ones(num_ris, 1);
assign = zeros(num_users, 1);  % 0表示未分配
assigned = false(num_users, 1);

% 迭代K轮，每轮分配一个用户
for iter = 1:num_users
    best_k = 0;
    best_l = 0;
    best_cost = inf;
    
    % 遍历所有未分配的用户
    for k = 1:num_users
        if assigned(k)
            continue;
        end
        
        % 检查直连选项
        if cost(k, 1) < best_cost
            best_k = k;
            best_l = 0;
            best_cost = cost(k, 1);
        end
        
        % 检查所有有容量的RIS
        for l = 1:num_ris
            if capacity(l) > 0 && cost(k, l + 1) < best_cost
                best_k = k;
                best_l = l;
                best_cost = cost(k, l + 1);
            end
        end
    end
    
    % 执行分配
    if best_k > 0
        assign(best_k) = best_l;
        assigned(best_k) = true;
        if best_l > 0
            capacity(best_l) = capacity(best_l) - 1;
        end
    end
end

end
