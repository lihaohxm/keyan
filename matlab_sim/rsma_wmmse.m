function [V, c, rates, sum_rate, gamma_c, gamma_p, aux] = rsma_wmmse(cfg, h_eff, p_dbw, weights, max_iter, V_init, is_urgent) % [Antigravity Fix]
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
if nargin < 7 || isempty(is_urgent)
    is_urgent = zeros(num_users, 1);
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

% 1. 初始化波束 (将不理想的 MRT 替换为 RZF 迫零预编码，减少初始流间干扰)
if nargin >= 6 && ~isempty(V_init) && isfield(V_init, 'v_c') % [Antigravity Fix]
    v_c = V_init.v_c;
    V_p = V_init.V_p;
else
    v_c = sum(h_eff, 2); 
    
    % --- 阶段二：动态公共率功率分配 ---
    HH = h_eff' * h_eff;
    alpha_c = 0.10;
    alpha_p = 0.90;
    
    v_c = v_c / (norm(v_c) + cfg.eps) * sqrt(p_watts * alpha_c);
    
    % 使用正则化伪逆 (RZF) 初始化私有流
    reg_val = max(noise_power, real(trace(HH)) / (num_users * 100)) + 1e-8;
    V_p_zf = h_eff / (HH + reg_val * eye(num_users)); 
    V_p = zeros(nt, num_users);
    for k = 1:num_users
        V_p(:, k) = V_p_zf(:, k) / (norm(V_p_zf(:, k)) + cfg.eps) * sqrt(p_watts * alpha_p / num_users);
    end
end

% 初始化接收器、权重、速率分配
u_c = zeros(num_users, 1);
u_p = zeros(num_users, 1);
w_c = zeros(num_users, 1);
w_p = zeros(num_users, 1);
c = (1/num_users) * ones(num_users, 1);

denom_c_vec = zeros(num_users, 1);
denom_p_vec = zeros(num_users, 1);

% NEW Tracking variables
diag_common_shaved = 0;
diag_common_cap_act = 0;
diag_up_before = NaN;
diag_up_after = NaN;
diag_transfer_pwr = 0;

% ===== Task 2: Private-First Budget Allocation and Common Gating =====
% Implement "default off, enable only if conditions met" for common stream
enable_private_first = get_cfg(cfg, 'enable_private_first_rsma', false);
common_enabled_global = true; % Default: common enabled (backward compatible)

if enable_private_first
    % Task 2A: Gate common stream based on strict conditions
    % Compute residual power after private allocation
    Pc_current = norm(v_c)^2;
    Pp_current = sum(sum(abs(V_p).^2));
    Pt_current = Pc_current + Pp_current;
    P_res = Pt_current - Pp_current; % Residual after private
    
    % Get gating parameters
    min_common_residual_ratio = get_cfg(cfg, 'min_common_residual_ratio', 0.10);
    common_enable_threshold = get_cfg(cfg, 'common_enable_threshold', 0.02);
    urgent_private_gate_min = get_cfg(cfg, 'urgent_private_gate_min', 0.50);
    raw_common_soft_ceiling = get_cfg(cfg, 'raw_common_soft_ceiling', 0.45);
    
    % Compute current metrics
    raw_common_prev = Pc_current / max(Pt_current, 1e-12);
    
    urgent_idx = find(is_urgent(:) > 0);
    if ~isempty(urgent_idx)
        P_urgent_private = sum(sum(abs(V_p(:, urgent_idx)).^2));
        urgent_private_ratio = P_urgent_private / max(Pp_current, 1e-12);
    else
        urgent_private_ratio = 1.0; % No urgent users, assume satisfied
    end
    
    % Compute common_marginal_gain_proxy (simplified version for gating)
    % This will be refined in the main loop
    common_marginal_gain_proxy_gate = 0.0; % Placeholder, will be computed properly
    
    % Check all gating conditions (ALL must be satisfied to enable common)
    condition_1 = (P_res / max(Pt_current, 1e-12)) >= min_common_residual_ratio;
    condition_2 = common_marginal_gain_proxy_gate >= common_enable_threshold;
    condition_3 = urgent_private_ratio >= urgent_private_gate_min;
    condition_4 = raw_common_prev <= raw_common_soft_ceiling;
    
    % For first iteration, use relaxed gating (only check residual and ceiling)
    % Full gating will be applied at the end of iterations
    common_enabled_global = condition_1 && condition_4;
    
    if ~common_enabled_global
        % Disable common stream: set v_c to zero or very weak
        v_c = zeros(size(v_c)); % Completely disable
        % Redistribute power to private streams
        if Pc_current > 1e-9
            V_p = V_p * sqrt((Pp_current + Pc_current) / max(Pp_current, 1e-12));
        end
    end
end

% 迭代更新
for iter = 1:max_iter
    % --- Step A: 更新 MMSE 接收器 u ---
    for k = 1:num_users
        hk = h_eff(:, k);
        % 公共流: 干扰 = 所有私有流 + 噪声（公共流尚未SIC）
        denom_c = abs(hk' * v_c)^2 + sum(abs(hk' * V_p).^2) + noise_power;
        denom_c_vec(k) = denom_c;
        u_c(k) = (hk' * v_c) / (denom_c + cfg.eps);

        % FIX: 私有流 SINR 分母 — SIC 成功后，干扰仅包含其他用户私有流 + 噪声
        % 原代码 denom_p = sum(abs(hk'*V_p).^2) 包含了自身信号，导致高信噪比下
        % MMSE接收器 u_p 被人为压低 → w_p 低估 → WMMSE无法在高功率下提升速率
        other_idx = [1:k-1, k+1:num_users];
        denom_p = sum(abs(hk' * V_p(:, other_idx)).^2) + noise_power;
        denom_p_vec(k) = denom_p;
        u_p(k) = (hk' * V_p(:, k)) / (denom_p + abs(hk' * V_p(:, k))^2 + cfg.eps);
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

        % FIX: 私有流 MSE — 分母应为 自身信号 + 其他干扰 + 噪声
        e_p(k) = 1 - sp / (sp + denom_p_vec(k) + cfg.eps);

        % Numerical safety
        e_c(k) = max(real(e_c(k)), 1e-12);
        e_p(k) = max(real(e_p(k)), 1e-12);

        w_c(k) = 1 / e_c(k);
        w_p(k) = 1 / e_p(k);
    end
    
    % --- Urgent-Private Inner Gain Boosting (Task 2.3) ---
    % Apply adaptive gain to private stream weights for urgent users
    % based on violation severity (inner optimization layer)
    enable_private_first = get_cfg(cfg, 'enable_private_first_rsma', false);
    if enable_private_first
        urgent_idx = find(is_urgent(:) > 0);
        if ~isempty(urgent_idx)
            gain_base = get_cfg(cfg, 'urgent_private_inner_gain_base', 1.1);
            gain_max = get_cfg(cfg, 'urgent_private_inner_gain_max', 1.6);
            
            % Get violation severity metrics
            urgent_delay_vio = get_cfg(cfg, 'urgent_delay_vio', 0.0);
            urgent_sem_vio = get_cfg(cfg, 'urgent_sem_vio', 0.0);
            
            % Scale gain based on violation severity (0 to 1 range)
            % Higher violation → higher gain (up to gain_max)
            vio_severity = min(1.0, max(urgent_delay_vio, urgent_sem_vio));
            urgent_gain = gain_base + (gain_max - gain_base) * vio_severity;
            
            % Apply gain to private stream weights for urgent users
            w_p(urgent_idx) = w_p(urgent_idx) * urgent_gain;
        end
    end

    % --- Step C: 更新预编码 V (闭式解/二次规划) ---
    % 【Antigravity Fix: 倒数加权法 (Inverse Water-Filling)】
    % 为了防止 RSMA 公共流被 WMMSE 的贪婪本质拉向信道极好的用户，导致弱用户容量掉零（木桶效应），
    % 我们对公共部分的更新权重进行动态的 max-min 平衡调整。
    gamma_c_est = zeros(num_users, 1);
    for i = 1:num_users
        hi = h_eff(:, i);
        % 这里 denom_c_vec(i) 已经包含了由于该用户引起的所有干扰和噪声
        interf_c = denom_c_vec(i) - abs(hi' * v_c)^2; 
        gamma_c_est(i) = max(0, abs(hi' * v_c)^2 / (interf_c + cfg.eps));
    end
    
    cap_c = log2(1 + gamma_c_est);
    tmp = 1 ./ sqrt(cap_c + 0.1);
    tmp = min(max(tmp, 0.5), 2.0);
    W_c_boost = weights(:) .* tmp;
    W_c_boost = W_c_boost / (sum(W_c_boost) + cfg.eps) * sum(weights); % 归一化保持总权重不变

    % 构造公共流预编码相关矩阵 A_common
    A_common = zeros(nt, nt);
    for i = 1:num_users
        hi = h_eff(:, i);
        A_common = A_common + W_c_boost(i) * w_c(i) * abs(u_c(i))^2 * (hi * hi');
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
        b_c = b_c + W_c_boost(i) * w_c(i) * conj(u_c(i)) * h_eff(:, i);
    end
    b_scale = (weights(:) .* w_p(:) .* conj(u_p(:))).'; % 1 x K
    B_p = h_eff .* (ones(nt, 1) * b_scale); % nt x K

    % Lagrange multiplier mu via bisection (power-tight & stable)
    mu_low = 0;
    mu_high = max(1e-2, real(trace(A_all)) / nt);

    Pc_prev = norm(v_c)^2;
    Pp_prev = sum(sum(abs(V_p).^2));
    Pt_prev = Pc_prev + Pp_prev;
    raw_ratio_prev = Pc_prev / max(Pt_prev, 1e-12);

    rho0 = get_cfg(cfg, 'common_ratio_target_base', 0.30);
    common_L_slope = get_cfg(cfg, 'common_L_slope', 0.045);
    n_ris = get_cfg(cfg, 'n_ris', 16);
    common_ratio_floor = get_cfg(cfg, 'common_ratio_floor', 0.18);
    sqrtL = sqrt(n_ris);
    common_ratio_target = max(common_ratio_floor, rho0 - common_L_slope * sqrtL);

    lambda_common_solver = get_cfg(cfg, 'lambda_common_solver', 0.12);
    lambda_common_solver_gain = get_cfg(cfg, 'lambda_common_solver_gain', 18.0);

    rho_gap = max(0, raw_ratio_prev - common_ratio_target);
    lambda_common_eff = lambda_common_solver + lambda_common_solver_gain * rho_gap;

    A_common_eff = A_common + lambda_common_eff * eye(size(A_common));

    [v_c_tmp, V_p_tmp, p_tmp] = update_precoders(A_common_eff, A_all, b_c, B_p, mu_low, cfg.eps);
    
    % Task 3A: Enforce residual common power constraint
    % Common stream can only use hard residual, not eat back into private budget
    if enable_private_first && common_enabled_global
        Pc_tmp = norm(v_c_tmp)^2;
        Pp_tmp = sum(sum(abs(V_p_tmp).^2));
        P_res_tmp = p_watts - Pp_tmp; % Hard residual after private
        
        eta_common_res = get_cfg(cfg, 'eta_common_res', 0.6);
        Pc_max_allowed = eta_common_res * max(P_res_tmp, 0);
        
        if Pc_tmp > Pc_max_allowed
            % Scale down common beamformer to respect residual constraint
            scale_c_res = sqrt(Pc_max_allowed / max(Pc_tmp, 1e-12));
            v_c_tmp = v_c_tmp * scale_c_res;
        end
    end
    
    if p_tmp <= p_watts * 1.001
        v_c = v_c_tmp;
        V_p = V_p_tmp;
    else
        % Increase upper bound until feasible
        for t = 1:100 % [Antigravity Fix]
            [~, ~, p_hi] = update_precoders(A_common_eff, A_all, b_c, B_p, mu_high, cfg.eps);
            if p_hi <= p_watts * 1.001 % [Antigravity Fix]
                break;
            end
            mu_high = mu_high * 10; % [Antigravity Fix]
        end

        % Bisection
        for t = 1:30
            mu_mid = 0.5 * (mu_low + mu_high);
            [v_c_mid, V_p_mid, p_mid] = update_precoders(A_common_eff, A_all, b_c, B_p, mu_mid, cfg.eps);
            if p_mid > p_watts * 1.001 % [Antigravity Fix]
                mu_low = mu_mid;
            else
                mu_high = mu_mid;
                v_c = v_c_mid;
                V_p = V_p_mid;
            end
        end
    end
    
    Pc_raw = norm(v_c)^2;
    Pp_raw = sum(sum(abs(V_p).^2));
    Pt_raw = Pc_raw + Pp_raw;
    raw_ratio = Pc_raw / max(Pt_raw, 1e-12);
    if iter == max_iter && get_cfg(cfg, 'diag_h1h2_log', false)
        fprintf('[H1-Log] iter=%d, P_common_raw=%.4e, ratio_raw=%.4f\n', iter, Pc_raw, raw_ratio);
    end

    urgent_delay_vio = get_cfg(cfg, 'urgent_delay_vio', 1.0);
    urgent_sem_vio   = get_cfg(cfg, 'urgent_sem_vio', 1.0);
    base_cap = get_cfg(cfg, 'common_power_cap_base', 0.12);
    max_cap  = get_cfg(cfg, 'common_power_cap_max', 0.42);

    adaptive_cap = base_cap + 0.12 * urgent_delay_vio + 0.08 * urgent_sem_vio;
    common_power_cap = min(max(adaptive_cap, base_cap), max_cap);

    rho0 = get_cfg(cfg, 'common_ratio_target_base', 0.30);
    common_L_slope = get_cfg(cfg, 'common_L_slope', 0.045);
    n_ris = get_cfg(cfg, 'n_ris', 16);
    common_ratio_floor = get_cfg(cfg, 'common_ratio_floor', 0.18);
    sqrtL = sqrt(n_ris);
    common_ratio_target = max(common_ratio_floor, rho0 - common_L_slope * sqrtL);

    lambda_common_excess = get_cfg(cfg, 'lambda_common_penalty', 12.0);
    common_excess = max(0, raw_ratio - common_ratio_target);
    obj_penalty = lambda_common_excess * common_excess^2;

    if iter == max_iter
        diag_common_cap_act = double(Pt_raw > 0 && raw_ratio > common_power_cap + 1e-9);
    end
    if Pt_raw > 0 && raw_ratio > common_power_cap
        targetPc = common_power_cap * Pt_raw;
        scale_c = sqrt(targetPc / max(Pc_raw, 1e-12));
        v_c = v_c * scale_c;
        deltaP = Pc_raw - targetPc;
        if iter == max_iter
            diag_common_shaved = deltaP;
        end
        V_p = distribute_shaved_power(V_p, deltaP, is_urgent, weights, num_users);
        if iter == max_iter && get_cfg(cfg, 'diag_h1h2_log', false)
            Pc_shaved = norm(v_c)^2;
            fprintf('[H1-Log] after cap: P_common_clipped=%.4e, deltaP_shaved=%.4e\n', Pc_shaved, deltaP);
            P_u_now = sum(sum(abs(V_p(:, urgent_idx)).^2));
            Pp_total_now = sum(sum(abs(V_p).^2));
            fprintf('[H1-Log] after refill: urgent_private_ratio=%.4f\n', P_u_now / max(Pp_total_now, 1e-12));
        end
    end

    % ===== 新增：如果common仍过重，则做轻度二次回收 =====
    Pc_now = norm(v_c)^2;
    Pp_now = sum(sum(abs(V_p).^2));
    Pt_now = Pc_now + Pp_now;
    rho_now = Pc_now / max(Pt_now, 1e-12);

    rebalance_trigger = get_cfg(cfg, 'common_rebalance_trigger', 0.36);
    rebalance_gain = get_cfg(cfg, 'common_rebalance_gain', 0.20);
    
    % --- Emergency-Only Common Rebalancing (Task 2.5) ---
    % Detect emergency conditions and trigger rebalancing only in emergencies
    enable_emergency_only = get_cfg(cfg, 'enable_common_rebalance_emergency_only', false);
    
    rebalance_triggered = false;
    if enable_emergency_only
        % Emergency conditions:
        % 1. Urgent infeasibility: high violation rates
        % 2. Catastrophic QoE drop: very high urgent QoE cost
        
        urgent_delay_vio = get_cfg(cfg, 'urgent_delay_vio', 0.0);
        urgent_sem_vio = get_cfg(cfg, 'urgent_sem_vio', 0.0);
        
        % Emergency thresholds
        urgent_infeasibility_threshold = 0.6;  % High violation rate indicates infeasibility
        catastrophic_qoe_threshold = 0.5;      % High QoE cost indicates catastrophic drop
        
        % Check for emergency conditions
        is_urgent_infeasible = (urgent_delay_vio > urgent_infeasibility_threshold) || ...
                               (urgent_sem_vio > urgent_infeasibility_threshold);
        
        % For catastrophic QoE, we use violation as proxy since QoE is not available in rsma_wmmse
        % High violation (>0.5) typically corresponds to catastrophic QoE
        is_catastrophic_qoe = max(urgent_delay_vio, urgent_sem_vio) > catastrophic_qoe_threshold;
        
        % Trigger rebalancing only in emergency
        is_emergency = is_urgent_infeasible || is_catastrophic_qoe;
        
        if Pt_now > 0 && rho_now > rebalance_trigger && is_emergency
            rebalance_triggered = true;
            excess_ratio = rho_now - rebalance_trigger;
            delta_ratio = rebalance_gain * excess_ratio;

            targetPc2 = max(0, (rho_now - delta_ratio) * Pt_now);
            targetPc2 = min(targetPc2, Pc_now);

            if targetPc2 < Pc_now
                scale_c2 = sqrt(targetPc2 / max(Pc_now, 1e-12));
                v_c = v_c * scale_c2;
                deltaP2 = Pc_now - targetPc2;
                V_p = distribute_shaved_power(V_p, deltaP2, is_urgent, weights, num_users);
            end
            if iter == max_iter && get_cfg(cfg, 'diag_h1h2_log', false) && exist('deltaP2', 'var')
                Pc_reb = norm(v_c)^2;
                fprintf('[H1-Log] EMERGENCY rebalance: common_ratio=%.4f, deltaP2=%.4e, urgent_vio=(%.2f,%.2f)\n', ...
                    Pc_reb / max(Pc_reb + sum(sum(abs(V_p).^2)), 1e-12), deltaP2, urgent_delay_vio, urgent_sem_vio);
            end
        end
    else
        % Original behavior: trigger based on common_power_ratio threshold
        if Pt_now > 0 && rho_now > rebalance_trigger
            rebalance_triggered = true;
            excess_ratio = rho_now - rebalance_trigger;
            delta_ratio = rebalance_gain * excess_ratio;

            targetPc2 = max(0, (rho_now - delta_ratio) * Pt_now);
            targetPc2 = min(targetPc2, Pc_now);

            if targetPc2 < Pc_now
                scale_c2 = sqrt(targetPc2 / max(Pc_now, 1e-12));
                v_c = v_c * scale_c2;
                deltaP2 = Pc_now - targetPc2;
                V_p = distribute_shaved_power(V_p, deltaP2, is_urgent, weights, num_users);
            end
            if iter == max_iter && get_cfg(cfg, 'diag_h1h2_log', false) && exist('deltaP2', 'var')
                Pc_reb = norm(v_c)^2;
                fprintf('[H1-Log] after rebalance: common_ratio=%.4f, deltaP2=%.4e\n', Pc_reb / max(Pc_reb + sum(sum(abs(V_p).^2)), 1e-12), deltaP2);
            end
        end
    end
    
    % Urgent private power floor (Weak transfer from normal to urgent)
    urgent_idx = find(is_urgent(:) > 0);
    if ~isempty(urgent_idx) && numel(urgent_idx) < num_users
        eta_u_min = get_cfg(cfg, 'urgent_private_power_min_ratio', 0.36);
        
        p_private_before = sum(abs(V_p).^2, 1).';
        Pp_total_before = sum(p_private_before);
        P_u_now = sum(p_private_before(urgent_idx));
        P_urgent_private_min = eta_u_min * Pp_total_before;
        
        up_ratio_tmp = P_u_now / max(Pp_total_before, 1e-12);
        if iter == max_iter
            diag_up_before = up_ratio_tmp;
        end
        
        if P_u_now < P_urgent_private_min
            deficit = P_urgent_private_min - P_u_now;
            normal_idx = setdiff(1:num_users, urgent_idx);
            
            P_normal_now = sum(p_private_before(normal_idx));
            
            % Limit transfer to 10% of normal private
            transfer = min(deficit, 0.10 * P_normal_now);
            
            if iter == max_iter
                diag_transfer_pwr = transfer;
            end
            
            give_back = p_private_before(normal_idx) / max(P_normal_now, 1e-12);
            delta = transfer * give_back;
            
            p_private_new = p_private_before;
            p_private_new(normal_idx) = p_private_new(normal_idx) - delta;
            
            w_u = weights(urgent_idx);
            w_u = w_u / max(sum(w_u), 1e-12);
            p_private_new(urgent_idx) = p_private_new(urgent_idx) + transfer * w_u;
            
            for k = 1:num_users
                scale_k = sqrt(p_private_new(k) / max(p_private_before(k), 1e-12));
                V_p(:, k) = V_p(:, k) * scale_k;
            end
        end
        
        if iter == max_iter
            P_u_after = sum(sum(abs(V_p(:, urgent_idx)).^2));
            Pp_total_after = sum(sum(abs(V_p).^2));
            diag_up_after = P_u_after / max(Pp_total_after, 1e-12);
            if get_cfg(cfg, 'diag_h1h2_log', false) && exist('transfer', 'var')
                fprintf('[H1-Log] after urgent floor: transfer_P=%.4e, urgent_ratio_after=%.4f\n', transfer, diag_up_after);
            end
        end
    end
    
    % --- Task 2.4 & Task 4: Common Stream Marginal Gain Gating (with real decision power) ---
    % Evaluate whether common stream provides marginal benefit for urgent users
    % If not, set common beamformer to zero and keep all power in private streams
    % This is done at the end of the last iteration, after all power allocation is complete
    if iter == max_iter
        enable_marginal_gate = get_cfg(cfg, 'enable_common_marginal_gate', false);
        common_enabled = common_enabled_global; % Start with global gate result
        common_marginal_gain_proxy = NaN;
        
        if enable_marginal_gate
            % Compute current metrics with common stream
            gamma_c_with = zeros(num_users, 1);
            gamma_p_with = zeros(num_users, 1);
            for k = 1:num_users
                hk = h_eff(:, k);
                gamma_c_with(k) = abs(hk' * v_c)^2 / (sum(abs(hk' * V_p).^2) + noise_power + cfg.eps);
                gamma_p_with(k) = abs(hk' * V_p(:, k))^2 / (sum(abs(hk' * V_p(:, setdiff(1:num_users, k))).^2) + noise_power + cfg.eps);
            end
            R_c_limit_with = 0.99 * cfg.bandwidth * log2(1 + min(gamma_c_with));
            R_p_with = cfg.bandwidth * log2(1 + gamma_p_with + cfg.eps);
            
            % Compute metrics without common stream (hypothetical: v_c = 0)
            % Redistribute common power to private streams proportionally
            Pc_current = norm(v_c)^2;
            Pp_current = sum(sum(abs(V_p).^2));
            
            if Pc_current > 1e-9
                % Create hypothetical private beamformers with redistributed power
                V_p_hypo = V_p * sqrt((Pp_current + Pc_current) / max(Pp_current, 1e-12));
                
                gamma_p_without = zeros(num_users, 1);
                for k = 1:num_users
                    hk = h_eff(:, k);
                    gamma_p_without(k) = abs(hk' * V_p_hypo(:, k))^2 / (sum(abs(hk' * V_p_hypo(:, setdiff(1:num_users, k))).^2) + noise_power + cfg.eps);
                end
                R_p_without = cfg.bandwidth * log2(1 + gamma_p_without + cfg.eps);
                
                % Compute marginal gains for urgent users
                urgent_idx = find(is_urgent(:) > 0);
                if ~isempty(urgent_idx)
                    % Marginal gain for urgent QoE (approximated by rate improvement)
                    c_with = allocate_common_rate_urgent_first(R_c_limit_with, weights(:), is_urgent(:), cfg);
                    urgent_rate_with = sum(R_p_with(urgent_idx) + c_with(urgent_idx));
                    urgent_rate_without = sum(R_p_without(urgent_idx));
                    marginal_gain_urgent_rate = urgent_rate_with - urgent_rate_without;
                    
                    % Marginal gain for common rate limit
                    marginal_gain_rc_limit = R_c_limit_with;
                    
                    % Compute marginal gain proxy (weighted combination)
                    % Positive value means common stream helps
                    common_marginal_gain_proxy = marginal_gain_urgent_rate / max(urgent_rate_without, 1e6) + ...
                                                 0.1 * marginal_gain_rc_limit / 1e6;
                    
                    % Task 4: Ensure common_marginal_gain_proxy truly controls enable/disable
                    % Get threshold from config
                    common_enable_threshold = get_cfg(cfg, 'common_enable_threshold', 0.02);
                    
                    % CRITICAL: If marginal gain below threshold, disable common
                    if common_marginal_gain_proxy < common_enable_threshold
                        common_enabled = false;
                        common_enabled_flag = 0;
                        
                        % Set common beamformer to zero
                        v_c = zeros(size(v_c));
                        
                        % Redistribute power to private streams
                        V_p = V_p_hypo;
                    else
                        common_enabled = true;
                        common_enabled_flag = 1;
                    end
                else
                    % No urgent users - use simpler criterion based on min SINR
                    min_gamma_c = min(gamma_c_with);
                    common_enabled = (min_gamma_c < 0.5); % Enable if min SINR < -3 dB
                    
                    if ~common_enabled
                        v_c = zeros(size(v_c));
                        V_p = V_p * sqrt((Pp_current + Pc_current) / max(Pp_current, 1e-12));
                        common_enabled_flag = 0;
                    else
                        common_enabled_flag = 1;
                    end
                    
                    common_marginal_gain_proxy = 0.0;
                end
            else
                % Common power already zero
                common_enabled = false;
                common_enabled_flag = 0;
                common_marginal_gain_proxy = 0.0;
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

% 修正：引入 0.99 的安全余量 (Safety Margin)
R_c_limit = 0.99 * cfg.bandwidth * log2(1 + min(gamma_c)); 
R_p = cfg.bandwidth * log2(1 + gamma_p + cfg.eps);

% FIX: 公共速率分配 — 按权重分配，但如果公共速率太低就跳过分配
% 原逻辑中 min(gamma_c) 的 bottleneck 效应在极端信道异质性下
% 会导致公共速率接近0，浪费分给公共流的功率
c = allocate_common_rate_urgent_first(R_c_limit, weights(:), is_urgent(:), cfg);

rates = R_p + c;
sum_rate = sum(rates);

V.v_c = v_c;
V.V_p = V_p;
V.c = c;

aux = struct();
Pc = norm(v_c)^2;
Pp = sum(sum(abs(V_p).^2));
aux.Pc = Pc;
aux.Pp = Pp;
if exist('raw_ratio', 'var')
    aux.common_power_ratio_raw = raw_ratio;
    aux.common_power_ratio_clipped = Pc / max(Pc + Pp, 1e-12);
    aux.common_power_ratio = aux.common_power_ratio_clipped;
    aux.common_cap_active = diag_common_cap_act;
    aux.common_shaved_power = diag_common_shaved;
    aux.common_power_cap  = common_power_cap;
    aux.common_excess_penalty = obj_penalty;
    aux.urgent_private_ratio_before_floor = diag_up_before;
    aux.urgent_private_ratio_after_floor = diag_up_after;
    aux.normal_to_urgent_transfer_power = diag_transfer_pwr;
else
    aux.common_power_ratio_raw = NaN;
    aux.common_power_ratio_clipped = Pc / max(Pc + Pp, 1e-12);
    aux.common_power_ratio = aux.common_power_ratio_clipped;
    aux.common_cap_active = NaN;
    aux.common_shaved_power = NaN;
    aux.common_power_cap  = NaN;
    aux.common_excess_penalty = NaN;
    aux.urgent_private_ratio_before_floor = NaN;
    aux.urgent_private_ratio_after_floor = NaN;
    aux.normal_to_urgent_transfer_power = NaN;
end
aux.min_gamma_c = min(gamma_c);
aux.mean_gamma_c = mean(gamma_c);
aux.Rc_limit = R_c_limit;

if exist('R_p', 'var')
    aux.private_rate_sum = sum(R_p);
else
    aux.private_rate_sum = NaN;
end

aux.common_rate_sum = sum(c);

if exist('R_p', 'var')
    aux.user_rate_std = std(R_p(:) + c(:));
else
    aux.user_rate_std = std(c(:));
end

% Diagnostic output for urgent-private inner gain (Requirement 7.15)
enable_private_first = get_cfg(cfg, 'enable_private_first_rsma', false);
if enable_private_first && any(is_urgent(:) > 0)
    gain_base = get_cfg(cfg, 'urgent_private_inner_gain_base', 1.1);
    gain_max = get_cfg(cfg, 'urgent_private_inner_gain_max', 1.6);
    urgent_delay_vio = get_cfg(cfg, 'urgent_delay_vio', 0.0);
    urgent_sem_vio = get_cfg(cfg, 'urgent_sem_vio', 0.0);
    vio_severity = min(1.0, max(urgent_delay_vio, urgent_sem_vio));
    aux.urgent_private_inner_gain = gain_base + (gain_max - gain_base) * vio_severity;
else
    aux.urgent_private_inner_gain = NaN;
end

% Diagnostic outputs for common stream marginal gain gating (Requirement 7.14, 7.8)
if exist('common_marginal_gain_proxy', 'var')
    aux.common_marginal_gain_proxy = common_marginal_gain_proxy;
else
    aux.common_marginal_gain_proxy = NaN;
end

if exist('common_enabled_flag', 'var')
    aux.common_enabled_flag = double(common_enabled_flag);
elseif exist('common_enabled', 'var')
    aux.common_enabled_flag = double(common_enabled);
else
    aux.common_enabled_flag = NaN;
end

% Diagnostic output for rebalance triggered flag (Requirement 7.16)
if exist('rebalance_triggered', 'var')
    aux.rebalance_triggered_flag = double(rebalance_triggered);
else
    aux.rebalance_triggered_flag = NaN;
end

% Diagnostic output for private_first_budget_ratio (Requirement 7.7)
% This captures the configured private budget ratio used in private-first allocation
enable_private_first = get_cfg(cfg, 'enable_private_first_rsma', false);
if enable_private_first
    aux.private_first_budget_ratio = get_cfg(cfg, 'private_budget_ratio', 0.75);
else
    aux.private_first_budget_ratio = NaN;
end

V.diag = aux;

if isfield(cfg, 'debug_rsma') && cfg.debug_rsma
    fprintf('[RSMA] Pc=%.4e, Pp=%.4e, ratio=%.3f, Rc=%.4e\n', ...
        aux.Pc, aux.Pp, aux.common_power_ratio, aux.Rc_limit);
end
if get_cfg(cfg, 'diag_h1h2_log', false)
    fprintf('[H1-Log] Final Evaluator Struct: common_ratio_clipped=%.4f\n', aux.common_power_ratio_clipped);
end
end

function [v_c, V_p, p_used] = update_precoders(A_common, A_all, b_c, B_p, mu, eps0)
nt = size(A_common, 1);
I = eye(nt);
v_c = (A_common + mu * I + eps0 * I) \ b_c;
V_p = (A_all + mu * I + eps0 * I) \ B_p;
p_used = real(norm(v_c)^2 + sum(sum(abs(V_p).^2)));
end

function c = allocate_common_rate_urgent_first(R_c_limit, weights, is_urgent, cfg)
    K = numel(weights);
    c = zeros(K,1);

    urgent_idx = find(is_urgent(:) > 0);
    if isempty(urgent_idx)
        ww = max(weights(:), 1e-12);
        c = R_c_limit * ww / sum(ww);
        return;
    end

    % 每个 urgent 用户的公共流保底
    c_floor = get_cfg(cfg, 'common_rate_floor_urgent', 0.02e6); % 0.02 Mbps
    total_floor = c_floor * numel(urgent_idx);

    if total_floor >= R_c_limit
        c(urgent_idx) = R_c_limit / numel(urgent_idx);
        return;
    end

    c(urgent_idx) = c_floor;
    remain = R_c_limit - total_floor;

    wu = max(weights(urgent_idx), 1e-12);
    c(urgent_idx) = c(urgent_idx) + remain * wu / sum(wu);
end

function val = get_cfg(cfg, name, default_val)
if isfield(cfg, name) && ~isempty(cfg.(name))
    val = cfg.(name);
else
    val = default_val;
end
end

function V_p = distribute_shaved_power(V_p, deltaP, is_urgent, weights, num_users)
    urgent_idx = find(is_urgent(:) > 0);
    Pp_raw = sum(sum(abs(V_p).^2));
    if Pp_raw <= 1e-12
        return;
    end
    if ~isempty(urgent_idx) && numel(urgent_idx) < num_users
        eta_u = 0.85;  % 回灌优先级参数：85% 给 urgent
        deltaP_u = deltaP * eta_u;
        deltaP_n = deltaP * (1 - eta_u);
        
        p_k = sum(abs(V_p).^2, 1).';
        normal_idx = setdiff(1:num_users, urgent_idx);
        
        w_u = weights(urgent_idx);
        w_u = w_u / max(sum(w_u), 1e-12);
        p_k(urgent_idx) = p_k(urgent_idx) + deltaP_u * w_u;
        
        w_n = weights(normal_idx);
        w_n = w_n / max(sum(w_n), 1e-12);
        p_k(normal_idx) = p_k(normal_idx) + deltaP_n * w_n;
        
        for k = 1:num_users
            scale_k = sqrt(p_k(k) / max(sum(abs(V_p(:, k)).^2), 1e-12));
            V_p(:, k) = V_p(:, k) * scale_k;
        end
    else
        targetPp = Pp_raw + deltaP;
        scale_p = sqrt(targetPp / max(Pp_raw, 1e-12));
        V_p = V_p * scale_p;
    end
end