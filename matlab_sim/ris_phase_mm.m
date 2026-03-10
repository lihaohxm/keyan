function [theta_new, info] = ris_phase_mm(cfg, ch, assign, V, weights, max_iter, theta_init)
%RIS_PHASE_MM RIS phase update via UQP + MM inner iterations.
% Paper mapping (Sec. 3.3, concise):
% 1) For RIS-l, write hk = hd_k + M_kl*theta_l under fixed (assign, V).
% 2) Build proxy objective as weighted WMMSE proxy (common + private).
% 3) FIX: Include interference suppression to non-assigned users.
% 4) This gives a unit-modulus quadratic program: minimize theta^H U theta - 2Re{v^H theta}.
% 5) MM majorization with lambda=max eig(U):
%    theta^{t+1} = exp(j*angle((lambda*I-U)*theta^t + v)).
% 6) If non-monotone, apply damping/rollback to preserve non-worsening proxy objective.
% 7) Unit-modulus is enforced every inner iteration by projection exp(j*angle(.)) elementwise.

num_ris = cfg.num_ris;
n_ris = cfg.n_ris;
K = cfg.num_users;

if nargin < 6 || isempty(max_iter)
    max_iter = get_cfg(cfg, 'ris_mm_iter', 6);
end
if nargin < 7
    theta_init = [];
end

mm_tol = get_cfg(cfg, 'ris_mm_tol', 1e-9);

% Phase 4: Bottleneck-Oriented RIS Proxy Configuration (Requirements 11.10-11.13)
enable_bottleneck_ris_proxy = get_cfg(cfg, 'enable_bottleneck_ris_proxy', false);
ris_common_term_scale_largeL = get_cfg(cfg, 'ris_common_term_scale_largeL', 0.2);
ris_bottleneck_beta = get_cfg(cfg, 'ris_bottleneck_beta', 2.0);
ris_bottleneck_mode = get_cfg(cfg, 'ris_bottleneck_mode', 'worst_m');
enable_structured_theta_starts = get_cfg(cfg, 'enable_structured_theta_starts', false);
enable_conditional_theta_damping = get_cfg(cfg, 'enable_conditional_theta_damping', false);

if isfield(cfg, 'ris_mm_alpha') && ~isempty(cfg.ris_mm_alpha)
    alpha0 = cfg.ris_mm_alpha;
else
    if cfg.n_ris >= 64
        alpha0 = 0.12;
    elseif cfg.n_ris >= 49
        alpha0 = 0.16;
    elseif cfg.n_ris >= 36
        alpha0 = 0.24;
    else
        alpha0 = 0.40;
    end
end

alpha_min = get_cfg(cfg, 'ris_mm_alpha_min', 1/64);
private_scale = get_cfg(cfg, 'ris_mm_private_scale', 1.0);
debug_mm = logical(get_cfg(cfg, 'ris_mm_debug', false));

if isempty(weights)
    weights = ones(K, 1);
end
weights = weights(:);
if numel(weights) ~= K
    error('ris_phase_mm:weights_size', 'weights must be Kx1.');
end

if ~isempty(theta_init)
    theta_new = theta_init;
else
    theta_new = ch.theta;
end

if size(theta_new, 1) ~= n_ris || size(theta_new, 2) ~= num_ris
    theta_new = ch.theta;
end

affect_users = cell(num_ris, 1);
obj_history_by_ris = cell(num_ris, 1);
mono_fail_by_ris = zeros(num_ris, 1);
mm_start_count_by_ris = zeros(num_ris, 1);
mm_best_obj_by_ris = zeros(num_ris, 1);
mm_final_obj_by_ris = zeros(num_ris, 1);
mm_obj_gain_by_ris = zeros(num_ris, 1);
mm_start_type_best = cell(num_ris, 1);  % Track which start type was best
mm_bottleneck_focus_weight = zeros(num_ris, 1);  % Track bottleneck weighting

% Get urgent user information if available
is_urgent = [];
if isfield(cfg, 'is_urgent') && ~isempty(cfg.is_urgent)
    is_urgent = cfg.is_urgent(:);
end

for l = 1:num_ris
    users_l = find(assign(:) == l);
    affect_users{l} = users_l;
    if isempty(users_l)
        obj_history_by_ris{l} = [];
        continue;
    end

    % FIX: 传入所有用户以考虑 RIS l 对非关联用户的干扰泄漏
    [U, v, bottleneck_weight] = build_uqp_terms_for_ris(cfg, ch, V, weights, assign, l, private_scale, ...
        enable_bottleneck_ris_proxy, ris_bottleneck_mode, ris_bottleneck_beta, ...
        ris_common_term_scale_largeL, is_urgent);
    
    mm_bottleneck_focus_weight(l) = bottleneck_weight;

    lambda_mm = max(real(eig((U + U') * 0.5)));
    if ~isfinite(lambda_mm)
        lambda_mm = 0;
    end

    % Structured multi-start initialization
    if enable_structured_theta_starts
        num_starts = get_cfg(cfg, 'ris_mm_num_starts', 6);
    else
        num_starts = get_cfg(cfg, 'ris_mm_num_starts', 1);
    end
    
    best_theta_l = theta_new(:, l);
    obj_init_l = proxy_objective(best_theta_l, U, v);
    best_obj_final = inf;
    best_obj_hist = [];
    best_start_type = 'current';

    for s = 1:num_starts
        if enable_structured_theta_starts
            % Structured initialization types
            if s == 1
                theta_l = theta_new(:, l);  % Current theta
                start_type = 'current';
            elseif s == 2
                % Small perturbation
                theta_l = theta_new(:, l) .* exp(1j * 0.1 * randn(n_ris, 1));
                theta_l = exp(1j * angle(theta_l));
                start_type = 'perturbed';
            elseif s == 3
                % Quantized theta (8 phases)
                angles = angle(theta_new(:, l));
                quantized = round(angles / (pi/4)) * (pi/4);
                theta_l = exp(1j * quantized);
                start_type = 'quantized';
            elseif s == 4
                % Phase-inverted
                theta_l = exp(1j * (angle(theta_new(:, l)) + pi));
                start_type = 'inverted';
            elseif s == 5
                % Urgent-private-oriented (maximize urgent private channel gains)
                if ~isempty(is_urgent) && any(is_urgent)
                    urgent_users_l = intersect(users_l, find(is_urgent));
                    if ~isempty(urgent_users_l)
                        % Initialize to maximize urgent user private channels
                        theta_l = zeros(n_ris, 1);
                        for uu = 1:numel(urgent_users_l)
                            k = urgent_users_l(uu);
                            M_kl = cfg.ris_gain * ch.G(:, :, l) * diag(ch.H_ris(:, k, l));
                            vk = V.V_p(:, k);
                            bp = M_kl' * vk;
                            theta_l = theta_l + bp / (norm(bp) + cfg.eps);
                        end
                        theta_l = exp(1j * angle(theta_l));
                        start_type = 'urgent_private';
                    else
                        theta_l = exp(1j * 2 * pi * rand(n_ris, 1));
                        start_type = 'random';
                    end
                else
                    theta_l = exp(1j * 2 * pi * rand(n_ris, 1));
                    start_type = 'random';
                end
            else
                % Random fallback
                theta_l = exp(1j * 2 * pi * rand(n_ris, 1));
                start_type = 'random';
            end
        else
            % Original behavior
            if s == 1
                theta_l = theta_new(:, l);
                start_type = 'current';
            else
                theta_l = exp(1j * 2 * pi * rand(n_ris, 1));
                start_type = 'random';
            end
        end

        obj_hist = zeros(max_iter + 1, 1);
        obj_prev = proxy_objective(theta_l, U, v);
        obj_hist(1) = obj_prev;
        
        % Track proxy improvement for conditional damping
        proxy_improvements = zeros(max_iter, 1);

        for it = 1:max_iter
            grad_vec = (lambda_mm * eye(n_ris) - U) * theta_l + v;
            theta_proj = exp(1j * angle(grad_vec + cfg.eps));

            obj_cand = proxy_objective(theta_proj, U, v);
            
            % Conditional theta damping
            if enable_conditional_theta_damping && it > 1
                % Check if proxy improvement is monotone
                recent_improvements = proxy_improvements(max(1, it-3):it-1);
                recent_improvements = recent_improvements(recent_improvements ~= 0);
                
                if ~isempty(recent_improvements) && all(recent_improvements > 0)
                    % Clear monotone improvement - use aggressive step
                    alpha_adaptive = min(1.0, alpha0 * 1.5);
                else
                    % Non-monotone - use conservative step
                    alpha_adaptive = alpha0;
                end
            else
                alpha_adaptive = alpha0;
            end
            
            if obj_cand <= obj_prev + mm_tol
                theta_try = theta_proj;
                obj_try = obj_cand;
                accepted = true;
            else
                accepted = false;
                alpha = min(1, max(alpha_adaptive, alpha_min));
                theta_try = theta_l;
                obj_try = obj_prev;
                while alpha >= alpha_min
                    theta_damped = exp(1j * angle((1 - alpha) * theta_l + alpha * theta_proj));
                    obj_damped = proxy_objective(theta_damped, U, v);
                    if obj_damped <= obj_prev + mm_tol
                        theta_try = theta_damped;
                        obj_try = obj_damped;
                        accepted = true;
                        break;
                    end
                    alpha = alpha * 0.5;
                end
            end

            if ~accepted
                mono_fail_by_ris(l) = mono_fail_by_ris(l) + 1;
                obj_hist(it + 1) = obj_prev;
                proxy_improvements(it) = 0;
                if debug_mm
                    fprintf('[ris-mm][l=%d][s=%d][it=%d] no-accept, keep obj=%.6e\n', l, s, it, obj_prev);
                end
                break;
            end

            theta_l = theta_try;
            obj_hist(it + 1) = obj_try;
            proxy_improvements(it) = obj_prev - obj_try;
            
            if abs(obj_prev - obj_try) <= mm_tol
                break;
            end
            obj_prev = obj_try;
        end

        if obj_prev < best_obj_final
            best_obj_final = obj_prev;
            best_theta_l = theta_l;
            best_obj_hist = obj_hist;
            best_start_type = start_type;
        end
    end

    last_nz = find(best_obj_hist ~= 0, 1, 'last');
    if isempty(last_nz)
        last_nz = 1;
    end
    obj_history_by_ris{l} = best_obj_hist(1:last_nz);
    theta_new(:, l) = best_theta_l;
    
    mm_start_count_by_ris(l) = num_starts;
    mm_best_obj_by_ris(l) = best_obj_final;
    mm_final_obj_by_ris(l) = best_obj_final;
    mm_obj_gain_by_ris(l) = obj_init_l - best_obj_final;
    mm_start_type_best{l} = best_start_type;
end

info = struct();
info.users_by_ris = affect_users;
info.obj_history_by_ris = obj_history_by_ris;
info.monotone_fail_count_by_ris = mono_fail_by_ris;

if exist('mm_start_count_by_ris', 'var')
    info.mm_start_count = mean(mm_start_count_by_ris);
    info.mm_best_obj = mean(mm_best_obj_by_ris);
    info.mm_final_obj = mean(mm_final_obj_by_ris);
    info.mm_obj_gain = mean(mm_obj_gain_by_ris);
    info.mm_alpha0 = alpha0;
    info.mm_interf_suppress = get_cfg(cfg, 'ris_interf_suppress', 0.15);
    info.mm_start_type_best = mm_start_type_best;  % Diagnostic: which start type was best
    info.mm_bottleneck_focus_weight = mean(mm_bottleneck_focus_weight);  % Diagnostic: bottleneck weighting
end
end

function [U, v, bottleneck_weight] = build_uqp_terms_for_ris(cfg, ch, V, weights, assign, l, private_scale, ...
    enable_bottleneck, bottleneck_mode, bottleneck_beta, common_scale_largeL, is_urgent)
% FIX: 原代码只对关联到 RIS l 的用户 (users_l) 构建信号增强项，
% 完全忽略了 RIS l 的相位变化对其他用户（非关联用户）产生的干扰。
% 当 RIS 数量增多时，这种未被控制的交叉干扰导致性能暴跌。
%
% 修复：对 RIS l 的代理目标函数 = Σ_{k∈users_l} wk * 有用信号功率
%                               - β * Σ_{k∉users_l} wk * 干扰泄漏功率
% 其中 β 是干扰抑制权重（默认=0.5），平衡信号增强与干扰抑制。
%
% Phase 4 Enhancement: Bottleneck-Oriented RIS Proxy
% - Focus on worst-case urgent users instead of mean enhancement
% - Adaptive common term scaling for large L
% - Three modes: worst_m, soft_bottleneck, dual_term

N = cfg.n_ris;
K = cfg.num_users;
U = zeros(N, N);
v = zeros(N, 1);

users_l = find(assign(:) == l);
interf_suppress = get_cfg(cfg, 'ris_interf_suppress', 0.15);

bottleneck_weight = 0;  % Track bottleneck focus weight

% Determine user weighting strategy
if enable_bottleneck && ~isempty(is_urgent)
    urgent_users_l = intersect(users_l, find(is_urgent));
    
    if ~isempty(urgent_users_l)
        % Bottleneck-oriented weighting
        user_weights = weights;
        
        switch lower(bottleneck_mode)
            case 'worst_m'
                % Mode 1: Focus only on m worst urgent users
                m_worst = max(1, floor(numel(urgent_users_l) / 2));
                [~, sorted_idx] = sort(weights(urgent_users_l), 'descend');
                worst_urgent = urgent_users_l(sorted_idx(1:m_worst));
                
                % Zero out non-worst users
                user_weights = zeros(K, 1);
                user_weights(worst_urgent) = weights(worst_urgent) * bottleneck_beta;
                bottleneck_weight = bottleneck_beta;
                
            case 'soft_bottleneck'
                % Mode 2: Exponential weighting (higher weight to worse users)
                % Assume higher weight = worse QoE
                user_weights = weights;
                user_weights(urgent_users_l) = weights(urgent_users_l) .^ bottleneck_beta;
                bottleneck_weight = mean(user_weights(urgent_users_l) ./ weights(urgent_users_l));
                
            case 'dual_term'
                % Mode 3: Combine urgent bottleneck + average private support
                user_weights = weights;
                user_weights(urgent_users_l) = weights(urgent_users_l) * bottleneck_beta;
                bottleneck_weight = bottleneck_beta;
                
            otherwise
                user_weights = weights;
        end
    else
        user_weights = weights;
    end
else
    user_weights = weights;
end

% Adaptive common term scaling for large L
common_weight_scale = 1.0;
if cfg.n_ris >= 49
    common_weight_scale = common_scale_largeL;
end

% Check if common power ratio is too high
common_power_ratio_threshold = 0.35;
if isfield(V, 'diag') && isfield(V.diag, 'common_power_ratio_raw')
    if V.diag.common_power_ratio_raw > common_power_ratio_threshold
        common_weight_scale = 0.0;  % Disable common enhancement
    end
end

% 1) 对关联用户: 最大化有用信号 (公共+私有)
for ii = 1:numel(users_l)
    k = users_l(ii);
    wk = user_weights(k);

    M_kl = cfg.ris_gain * ch.G(:, :, l) * diag(ch.H_ris(:, k, l));
    h0 = ch.h_d(:, k);

    % Common stream with adaptive scaling
    bc = M_kl' * V.v_c;
    cc = h0' * V.v_c;
    U = U - wk * common_weight_scale * (bc * bc');
    v = v + wk * common_weight_scale * conj(cc) * bc;

    % Private stream
    vk = V.V_p(:, k);
    bp = M_kl' * vk;
    cp = h0' * vk;
    U = U - wk * private_scale * (bp * bp');
    v = v + wk * private_scale * conj(cp) * bp;
end

% 2) FIX: 对非关联用户: 抑制干扰泄漏
% RIS l 反射信号会泄漏到非关联用户，增加他们的干扰
if interf_suppress > 0
    other_users = setdiff(1:K, users_l);
    for ii = 1:numel(other_users)
        k = other_users(ii);
        wk = weights(k);
        
        M_kl = cfg.ris_gain * ch.G(:, :, l) * diag(ch.H_ris(:, k, l));
        
        for j = 1:size(V.V_p, 2)
            if j == k, continue; end
            bj = M_kl' * V.V_p(:, j);
            U = U + interf_suppress * wk * (bj * bj');
        end
    end
end

U = (U + U') * 0.5;
end

function obj = proxy_objective(theta, U, v)
obj = real(theta' * U * theta - 2 * real(v' * theta));
end

function val = get_cfg(cfg, name, default_val)
if isfield(cfg, name) && ~isempty(cfg.(name))
    val = cfg.(name);
else
    val = default_val;
end
end
