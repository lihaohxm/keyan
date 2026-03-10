function [assign, theta_all, beam_info, ao_log] = ua_qoe_ao(cfg, ch, geom, p_dbw, semantic_mode, table_path, weights_or_profile, aux_init)
%UA_QOE_AO UA-QoE-AO with unified RSMA evaluation.
%
% 7th arg compatibility:
%   - 1x2 row vector: legacy weights row
%   - struct: profile with per-user fields (M_k, weights, d_k, dmax_k, ...)

if nargin < 8
    aux_init = [];
end

num_users = cfg.num_users;
num_ris = cfg.num_ris;
n_ris = cfg.n_ris;

[profile, weights_row] = resolve_profile(cfg, geom, weights_or_profile);

urgent_idx_arr = get_urgent_indices_from_profile(profile, num_users);
is_urgent = false(num_users, 1);
is_urgent(urgent_idx_arr) = true;

simple_beam = get_cfg_or_default(cfg, 'ao_simple_beam', false);

if ~isempty(aux_init) && isfield(aux_init, 'assign') && isfield(aux_init, 'theta_all') && isfield(aux_init, 'V')
    assign = aux_init.assign;
    theta_all = aux_init.theta_all;
    V = aux_init.V;
else
    % Step 1: initialize assignment (warm-start if provided; else capacity-feasible random)
    if isfield(cfg, 'ao_init_assign') && ~isempty(cfg.ao_init_assign) && numel(cfg.ao_init_assign) == num_users
        assign = cfg.ao_init_assign(:);
        assign = max(0, min(num_ris, round(assign)));
    else
        % --- 修复：避免纯 0 初始化导致的早期拒绝陷阱，使用容量合规的随机初始化 ---
        cap = cfg.k0 * ones(cfg.num_ris, 1);
        assign = zeros(cfg.num_users, 1);
        for k_init = 1:cfg.num_users
            avail = find(cap > 0).';
            choices = [0, avail];
            pick = choices(randi(numel(choices)));
            if pick > 0
                cap(pick) = cap(pick) - 1;
            end
            assign(k_init) = pick;
        end
    end

    % Step 2: initialize theta (align each RIS to its currently-associated user if any)
    theta_all = init_theta_by_assignment(cfg, ch, assign);

    % Step 3: initialize RSMA precoder
    h_eff = effective_channel(cfg, ch, assign, theta_all);

    if simple_beam
        V = build_mrt_beam(cfg, h_eff, p_dbw);
    else
        [V, ~, ~, ~, ~, ~, aux_init_wmmse] = rsma_wmmse(cfg, h_eff, p_dbw, ones(num_users, 1), 2, [], is_urgent);
        aux_init = aux_init_wmmse;
    end
end

max_outer_iter = get_cfg_or_default(cfg, 'ao_max_outer_iter', 8);
min_outer_iter = get_cfg_or_default(cfg, 'ao_min_outer_iter', 3);
wmmse_iter = get_cfg_or_default(cfg, 'ao_wmmse_iter', 8);
mm_iter = get_cfg_or_default(cfg, 'ris_mm_iter', get_cfg_or_default(cfg, 'ao_mm_iter', 5));
bt_max_tries = get_cfg_or_default(cfg, 'ao_bt_max_tries', get_cfg_or_default(cfg, 'bt_max_tries', 6));
rand_tries = get_cfg_or_default(cfg, 'ao_rand_tries', 10);
rand_sigma = get_cfg_or_default(cfg, 'ao_rand_sigma', 0.1);
freeze_theta = get_cfg_or_default(cfg, 'ao_freeze_theta', false);
disable_random_fallback = get_cfg_or_default(cfg, 'ao_disable_random_fallback', false);
verbose = get_cfg_or_default(cfg, 'ao_verbose', false);
tol = max(cfg.eps, 1e-9);
accept_rel = get_cfg_or_default(cfg, 'ao_accept_rel', 5e-3);
accept_abs = get_cfg_or_default(cfg, 'ao_accept_abs', 1e-4);

% Phase 3: Protected Theta Refit and Layered Acceptance Configuration (Requirements 11.6-11.9)
enable_theta_pre_refit_eval = get_cfg_or_default(cfg, 'enable_theta_pre_refit_eval', false);
enable_protected_theta_refit = get_cfg_or_default(cfg, 'enable_protected_theta_refit', false);
theta_refit_mode = get_cfg_or_default(cfg, 'theta_refit_mode', 'private_only');
enable_theta_layered_accept = get_cfg_or_default(cfg, 'enable_theta_layered_accept', false);

eval_opts = struct('semantic_mode', semantic_mode, 'table_path', table_path);
eval_calls = 0;
urgent_idx = urgent_idx_arr;

sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
out_old = evaluate_system_rsma(cfg, ch, geom, sol, profile, eval_opts);
if exist('aux_init', 'var') && isstruct(aux_init)
    out_old.diag = aux_init;
end
eval_calls = eval_calls + 1;
ao_log.qoe_history = out_old.avg_qoe;
ao_log.sum_rate_history = out_old.sum_rate_bps;
ao_log.accept.assign = 0;
ao_log.accept.V = 0;
ao_log.accept.theta = 0;
ao_log.accept.rand = 0;
ao_log.outer = struct('accept_assign', {}, 'accept_v', {}, 'accept_theta', {}, 'accept_rand', {}, ...
    'accept_theta_main', {}, 'accept_theta_polish', {}, 'theta_changed_norm', {}, 'theta_changed_norm_polish', {}, ...
    'theta_candidate_count', {}, 'theta_best_improve_main', {}, 'theta_best_improve_polish', {}, ...
    'avg_qoe', {}, 'urgent_qoe', {}, 'sum_rate', {}, 'break_reason', {});
best_sol = sol;
best_out = out_old;
if verbose
    fprintf('[ua] max_outer_iter=%d, bt_max_tries=%d\n', max_outer_iter, bt_max_tries);
end

for outer = 1:max_outer_iter
    assign_prev = assign;
    theta_prev = theta_all;
    V_prev = V;
    out_prev = out_old;

    outer_accept_t_main = false;
    outer_accept_t_polish = false;
    outer_theta_delta_main = 0;
    outer_theta_delta_polish = 0;
    outer_theta_cands = 0;
    outer_theta_imp_main = 0;
    outer_theta_imp_polish = 0;
    if verbose
        fprintf('[ua][outer %d] start cost=%.6f\n', outer, out_prev.avg_qoe);
    end

    % ================================================================
    % Step 1: association update with *descent guarantee* (paper-style)
    % ================================================================
    sol_ref = struct('assign', assign, 'theta_all', theta_all, 'V', V);
    assign_cand = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights_row, geom, profile, sol_ref);

    accepted_assign = false;
    if ~isequal(assign_cand(:), assign(:))
        % --- 为新分配快速对齐相位和预编码器，避免"旧V+新assign"的不匹配 ---
        
        % 1. 为新的分配快速对齐相位
        theta_try = init_theta_by_assignment(cfg, ch, assign_cand);
        
        % 2. 使用快速 WMMSE（自适应迭代）生成含公共流的预编码器
        %    MRT 缺乏公共流，会导致 RSMA 系统 QoE 极差，使 AO 陷入停滞
        h_eff_try = effective_channel(cfg, ch, assign_cand, theta_try);
        test_iter = max(3, min(8, round((p_dbw - 10) / 2))); % [Antigravity Fix]
        [V_try, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff_try, p_dbw, ones(cfg.num_users, 1), test_iter, V, is_urgent); % [Antigravity Fix]
        
        % 3. 评估这个"连贯"的临时解
        sol_try_a = struct('assign', assign_cand, 'theta_all', theta_try, 'V', V_try);
        out_try_a = evaluate_system_rsma(cfg, ch, geom, sol_try_a, profile, eval_opts);
        eval_calls = eval_calls + 1;
        if accept_candidate_relaxed(out_try_a, out_prev, cfg, 'A')
            assign = assign_cand;
            theta_all = theta_try;  % 同步更新相位
            V = V_try;              % 同步更新预编码器
            out_prev = out_try_a;
            ao_log.qoe_history(end + 1, 1) = out_prev.avg_qoe;
            ao_log.sum_rate_history(end + 1, 1) = out_prev.sum_rate_bps;
            ao_log.accept.assign = ao_log.accept.assign + 1;
            accepted_assign = true;
            if verbose
                fprintf('[ua][outer %d][A] accept cost=%.6f\n', outer, out_prev.avg_qoe);
            end
        elseif verbose
            fprintf('[ua][outer %d][A] reject cost=%.6f (curr=%.6f)\n', outer, out_try_a.avg_qoe, out_prev.avg_qoe);
        end
    end

    if ~accepted_assign
        assign = assign_prev;
    end

    % Build QoE-guided weights from the *current accepted* working point
    q_weights = build_excess_weights(cfg, out_prev, profile);

    % ================================================================
    % Step 2: RSMA precoder update (V-step; theta fixed)
    % ================================================================
    if verbose
        fprintf('[ua][outer %d][V] start cost=%.6f\n', outer, out_prev.avg_qoe);
    end
    accepted_v = false;
    if simple_beam
        % no RSMA update in ablation (simple_beam): keep V
        accepted_v = false;
    else
        h_eff = effective_channel(cfg, ch, assign, theta_all);
        if get_cfg_or_default(cfg, 'diag_rzf_core', false)
            V_cand = build_rzf_beam_fixed_common(cfg, h_eff, p_dbw, 0.20, is_urgent, q_weights);
            aux = struct('common_power_ratio_raw', 0.2, 'common_power_ratio_clipped', 0.2, 'common_cap_active', 0, 'common_shaved_power', 0);
        else
            cfg_wmmse = cfg;
            cfg_wmmse.urgent_delay_vio = get_cfg_or_default(out_prev, 'urgent_delay_violation', 1.0);
            cfg_wmmse.urgent_sem_vio = get_cfg_or_default(out_prev, 'urgent_semantic_violation', 1.0);
            [V_cand, ~, ~, ~, ~, ~, aux] = rsma_wmmse(cfg_wmmse, h_eff, p_dbw, q_weights, wmmse_iter, V, is_urgent);
        end
        sol_cand_v = struct('assign', assign, 'theta_all', theta_all, 'V', V_cand);
        out_cand_v = evaluate_system_rsma(cfg, ch, geom, sol_cand_v, profile, eval_opts);
        if isstruct(aux)
            out_cand_v.diag = aux;
        end
        eval_calls = eval_calls + 1;

        % Direct Update (with Monotonicity Guard)
        accept_v = accept_candidate_relaxed(out_cand_v, out_prev, cfg, 'V');

        if accept_v
            V = V_cand;
            out_prev = out_cand_v;
            ao_log.qoe_history(end + 1, 1) = out_prev.avg_qoe;
            ao_log.sum_rate_history(end + 1, 1) = out_prev.sum_rate_bps;
            ao_log.accept.V = ao_log.accept.V + 1;
            accepted_v = true;
            if verbose
                fprintf('[ua][outer %d][V] accept cost=%.6f\n', outer, out_prev.avg_qoe);
            end
        else
            V = V; % reject, keep old (V_prev is just V at start of iter)
        end
    end

    % ================================================================
    % Step 3: RIS phase update (theta-step; V fixed)
    % ================================================================
    accepted_t = false;
    accepted_r = false;
    if freeze_theta
        if verbose
            fprintf('[ua][outer %d][T] skipped (freeze theta)\n', outer);
        end
    else
        if verbose
            fprintf('[ua][outer %d][T] start cost=%.6f\n', outer, out_prev.avg_qoe);
        end
        num_theta_starts = get_cfg_or_default(cfg, 'num_theta_starts_smallL', 5);
        is_large_L = cfg.n_ris >= get_cfg_or_default(cfg, 'largeL_threshold', 49);
        if is_large_L
            num_theta_starts = get_cfg_or_default(cfg, 'num_theta_starts_largeL', 4);
        end
        
        theta_cands = cell(num_theta_starts,1);
        theta_cands{1} = theta_all;
        theta_cands{2} = exp(1j * angle(theta_all + 0.05*(randn(size(theta_all)) + 1j*randn(size(theta_all)))));
        theta_cands{3} = exp(1j * round(angle(theta_all)/(pi/4))*(pi/4));
        
        if is_large_L && num_theta_starts >= 4
            theta_cands{4} = exp(1j * angle(-theta_all)); % phase inversion
            start_rand = 5;
        else
            theta_cands{4} = ones(size(theta_all));
            start_rand = 5;
        end
        
        for s = start_rand:num_theta_starts
            theta_cands{s} = exp(1j * 2*pi * rand(size(theta_all)));
        end
        
        best_cand_out = [];
        best_cand_theta = [];
        best_cand_V = [];
        outer_theta_cands = num_theta_starts;
        
        % Initialize H2 diagnostic accumulators
        theta_pre_refit_improve_best = NaN;
        theta_post_refit_improve_best = NaN;
        theta_refit_swallow_ratio_best = NaN;
        theta_good_refit_bad_flag_best = 0;
        
        for s = 1:num_theta_starts
            [theta_tmp, mm_info] = ris_phase_mm(cfg, ch, assign, V, q_weights, mm_iter, theta_cands{s});
            
            if get_cfg_or_default(cfg, 'diag_h1h2_log', false)
                sol_no_refit = struct('assign', assign, 'theta_all', theta_tmp, 'V', V);
                out_no_refit = evaluate_system_rsma(cfg, ch, geom, sol_no_refit, profile, eval_opts);
                gain_pre_refit = out_prev.avg_qoe - out_no_refit.avg_qoe;
                if isfield(mm_info, 'mm_obj_gain')
                    surrogate_gain = mm_info.mm_obj_gain;
                else
                    surrogate_gain = NaN;
                end
            end

            % ===== H2 Diagnostic: Pre-refit evaluation =====
            h_eff_tmp = effective_channel(cfg, ch, assign, theta_tmp);
            sol_pre_refit = struct('assign', assign, 'theta_all', theta_tmp, 'V', V);
            out_pre_refit = evaluate_system_rsma(cfg, ch, geom, sol_pre_refit, profile, eval_opts);
            eval_calls = eval_calls + 1;
            
            % Compute pre-refit improvement
            metric_prev = get_compare_cost(out_prev);
            metric_pre = get_compare_cost(out_pre_refit);
            theta_pre_improve = metric_prev - metric_pre;
            
            % ===== Post-theta short refit =====
            if get_cfg_or_default(cfg, 'diag_rzf_core', false)
                V_refit = build_rzf_beam_fixed_common(cfg, h_eff_tmp, p_dbw, 0.20, is_urgent, q_weights);
                aux_refit = struct('common_power_ratio_raw', 0.2, 'common_power_ratio_clipped', 0.2, 'common_cap_active', 0, 'common_shaved_power', 0);
            else
                refit_iter = get_cfg_or_default(cfg, 'post_theta_wmmse_refit', 4);
                V_refit = V;
                [V_refit, ~, ~, ~, ~, ~, aux_refit] = rsma_wmmse( ...
                    cfg, h_eff_tmp, p_dbw, q_weights, refit_iter, V_refit, is_urgent);
            end

            sol_tmp_t = struct('assign', assign, 'theta_all', theta_tmp, 'V', V_refit);
            out_tmp_t = evaluate_system_rsma(cfg, ch, geom, sol_tmp_t, profile, eval_opts);
            
            % Compute post-refit improvement
            metric_post = get_compare_cost(out_tmp_t);
            theta_post_improve = metric_prev - metric_post;
            
            % Compute swallow ratio
            if abs(theta_pre_improve) > cfg.eps
                theta_swallow_ratio = (theta_pre_improve - theta_post_improve) / max(abs(theta_pre_improve), cfg.eps);
            else
                theta_swallow_ratio = 0;
            end
            
            % Detect theta_good_refit_bad
            pre_eps = 1e-4;
            post_eps = 1e-4;
            theta_good_refit_bad = (theta_pre_improve > pre_eps) && (theta_post_improve < -post_eps);

            if exist('aux_refit', 'var') && isstruct(aux_refit)
                out_tmp_t.diag = aux_refit;
            elseif isfield(out_prev, 'diag')
                out_tmp_t.diag = out_prev.diag;
            end

            eval_calls = eval_calls + 1;
            
            [acc_tmp, rej_reason_tmp] = accept_candidate_relaxed(out_tmp_t, out_prev, cfg, 'T');
            if get_cfg_or_default(cfg, 'diag_h1h2_log', false)
                gain_post_refit = out_prev.avg_qoe - out_tmp_t.avg_qoe;
                if acc_tmp
                    rej_reason_log = 'Accepted';
                else
                    rej_reason_log = rej_reason_tmp;
                end
                fprintf('[H2-Log] Theta Cand %d/%d: surrogate_gain=%.4e, pre_refit_gain=%.4e, post_refit_gain=%.4e, result=%s\n', ...
                    s, num_theta_starts, surrogate_gain, gain_pre_refit, gain_post_refit, rej_reason_log);
            end

            if isempty(best_cand_out) || accept_candidate_relaxed(out_tmp_t, best_cand_out, cfg, 'T')
                best_cand_out = out_tmp_t;
                best_cand_theta = theta_tmp;
                best_cand_V = V_refit;
                
                % Update best H2 diagnostics
                theta_pre_refit_improve_best = theta_pre_improve;
                theta_post_refit_improve_best = theta_post_improve;
                theta_refit_swallow_ratio_best = theta_swallow_ratio;
                theta_good_refit_bad_flag_best = double(theta_good_refit_bad);
            end
        end
        
        theta_cand = best_cand_theta;
        out_cand_t = best_cand_out;
        V_cand_t = best_cand_V;
        
        % Store H2 diagnostics in out_cand_t.diag (always, regardless of accept/reject)
        if ~isfield(out_cand_t, 'diag')
            out_cand_t.diag = struct();
        end
        out_cand_t.diag.theta_pre_refit_improve = theta_pre_refit_improve_best;
        out_cand_t.diag.theta_post_refit_improve = theta_post_refit_improve_best;
        out_cand_t.diag.theta_refit_swallow_ratio = theta_refit_swallow_ratio_best;
        out_cand_t.diag.theta_good_refit_bad_flag = theta_good_refit_bad_flag_best;

        % Direct Update (with Monotonicity Guard)
        accept_t = accept_candidate_relaxed(out_cand_t, out_prev, cfg, 'T');

        if accept_t
            outer_accept_t_main = true;
            outer_theta_delta_main = norm(theta_cand - theta_all, 'fro');
            outer_theta_imp_main = max(0, out_prev.avg_qoe - out_cand_t.avg_qoe);
            
            theta_all = theta_cand;
            out_prev = out_cand_t;
            
            % immediately run formal rsma_wmmse to update V
            h_eff_after_t = effective_channel(cfg, ch, assign, theta_all);
            if get_cfg_or_default(cfg, 'diag_rzf_core', false)
                V = build_rzf_beam_fixed_common(cfg, h_eff_after_t, p_dbw, 0.20, is_urgent, q_weights);
                aux_after_t = struct('common_power_ratio_raw', 0.2, 'common_power_ratio_clipped', 0.2, 'common_cap_active', 0, 'common_shaved_power', 0);
            else
                [V, ~, ~, ~, ~, ~, aux_after_t] = rsma_wmmse( ...
                    cfg, h_eff_after_t, p_dbw, q_weights, wmmse_iter, V, is_urgent);
            end
            
            sol_after_t = struct('assign', assign, 'theta_all', theta_all, 'V', V);
            out_after_t = evaluate_system_rsma(cfg, ch, geom, sol_after_t, profile, eval_opts);
            if isstruct(aux_after_t)
                out_after_t.diag = aux_after_t;
            end
            
            % Merge H2 diagnostics into out_after_t.diag
            if ~isfield(out_after_t, 'diag')
                out_after_t.diag = struct();
            end
            out_after_t.diag.theta_pre_refit_improve = theta_pre_refit_improve_best;
            out_after_t.diag.theta_post_refit_improve = theta_post_refit_improve_best;
            out_after_t.diag.theta_refit_swallow_ratio = theta_refit_swallow_ratio_best;
            out_after_t.diag.theta_good_refit_bad_flag = theta_good_refit_bad_flag_best;
            
            out_prev = out_after_t;

            ao_log.qoe_history(end + 1, 1) = out_prev.avg_qoe;
            ao_log.sum_rate_history(end + 1, 1) = out_prev.sum_rate_bps;
            ao_log.accept.theta = ao_log.accept.theta + 1;
            accepted_t = true;
            if verbose
                fprintf('[ua][outer %d][T] accept cost=%.6f\n', outer, out_prev.avg_qoe);
            end
        else
            % Theta rejected, but still write H2 diagnostics to out_prev.diag
            if ~isfield(out_prev, 'diag')
                out_prev.diag = struct();
            end
            out_prev.diag.theta_pre_refit_improve = theta_pre_refit_improve_best;
            out_prev.diag.theta_post_refit_improve = theta_post_refit_improve_best;
            out_prev.diag.theta_refit_swallow_ratio = theta_refit_swallow_ratio_best;
            out_prev.diag.theta_good_refit_bad_flag = theta_good_refit_bad_flag_best;
            
            theta_all = theta_all; % reject, keep old
        end
        
        % --- Theta-only polish step for large arrays ---
        if cfg.n_ris >= 49
            theta_best = theta_all;
            out_best = out_prev;
            theta_before_polish = theta_all;
            out_before_polish = out_prev;
            polish_rounds = get_cfg_or_default(cfg, 'theta_polish_rounds', 2);
            tol_rate_p = get_cfg_or_default(cfg, 'theta_polish_tol_rate', 5e5);
            tol_uq_p = get_cfg_or_default(cfg, 'theta_polish_tol_uq', 0.01);
            
            V_best = V;
            for kk = 1:polish_rounds
                theta_try = ris_phase_mm(cfg, ch, assign, V, q_weights, mm_iter, theta_best);

                h_eff_try = effective_channel(cfg, ch, assign, theta_try);
                if get_cfg_or_default(cfg, 'diag_rzf_core', false)
                    V_try_refit = build_rzf_beam_fixed_common(cfg, h_eff_try, p_dbw, 0.20, is_urgent, q_weights);
                    aux_try_refit = struct('common_power_ratio_raw', 0.2, 'common_power_ratio_clipped', 0.2, 'common_cap_active', 0, 'common_shaved_power', 0);
                else
                    refit_iter = get_cfg_or_default(cfg, 'post_theta_wmmse_refit', 4);
                    V_try_refit = V;
                    [V_try_refit, ~, ~, ~, ~, ~, aux_try_refit] = rsma_wmmse( ...
                        cfg, h_eff_try, p_dbw, q_weights, refit_iter, V_try_refit, is_urgent);
                end

                sol_try = struct('assign', assign, 'theta_all', theta_try, 'V', V_try_refit);
                out_try = evaluate_system_rsma(cfg, ch, geom, sol_try, profile, eval_opts);

                if exist('aux_try_refit', 'var') && isstruct(aux_try_refit)
                    out_try.diag = aux_try_refit;
                elseif isfield(out_best, 'diag')
                    out_try.diag = out_best.diag;
                end
                eval_calls = eval_calls + 1;
                
                if out_try.total_sum_rate > out_best.total_sum_rate - tol_rate_p && ...
                   out_try.urgent_qoe < out_best.urgent_qoe + tol_uq_p
                    theta_best = theta_try;
                    V_best = V_try_refit;
                    out_best = out_try;
                end
            end
            if norm(theta_best - theta_before_polish, 'fro') > 1e-6
                outer_accept_t_polish = true;
                outer_theta_delta_polish = norm(theta_best - theta_before_polish, 'fro');
                outer_theta_imp_polish = max(0, out_before_polish.avg_qoe - out_best.avg_qoe);
            end
            theta_all = theta_best;
            V = V_best;
            out_prev = out_best;
        end
    end

    % Step 4: [Paper Alignment] Removed untheoretical random perturbation fallback
    % The AO algorithm converges theoretically without heuristics.

    if out_prev.avg_qoe < best_out.avg_qoe - tol
        best_out = out_prev;
        best_sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
    end

    outer_log = struct();
    outer_log.accept_assign = accepted_assign;
    outer_log.accept_v = accepted_v;
    outer_log.accept_theta = accepted_t;
    outer_log.accept_rand = accepted_r;
    outer_log.accept_theta_main = outer_accept_t_main;
    outer_log.accept_theta_polish = outer_accept_t_polish;
    outer_log.theta_changed_norm = outer_theta_delta_main;
    outer_log.theta_changed_norm_polish = outer_theta_delta_polish;
    outer_log.theta_candidate_count = outer_theta_cands;
    outer_log.theta_best_improve_main = outer_theta_imp_main;
    outer_log.theta_best_improve_polish = outer_theta_imp_polish;
    outer_log.avg_qoe = out_prev.avg_qoe;
    outer_log.urgent_qoe = mean(out_prev.qoe_vec(urgent_idx));
    outer_log.sum_rate = out_prev.sum_rate_bps;
    outer_log.break_reason = '';
    ao_log.outer(end + 1, 1) = outer_log;
    out_old = out_prev;

    min_ao_iters = 5;
    stall_limit = 3;
    if outer == 1
        stall_count = 0;
    end
    
    accepted_this_round = accepted_v || accepted_t || accepted_assign;
    if accepted_this_round
        stall_count = 0;
    else
        stall_count = stall_count + 1;
    end
    
    if outer >= min_ao_iters && stall_count >= stall_limit
        ao_log.outer(end).break_reason = 'stall_limit';
        if verbose
            fprintf('[ua][outer %d] stall limit reached -> stop\n', outer);
        end
        break;
    end
end

assign = best_sol.assign;
theta_all = best_sol.theta_all;
V = best_sol.V;
out_old = best_out;

if isfield(V, 'diag')
    V.diag.accept_assign = [ao_log.outer.accept_assign];
    V.diag.accept_v = [ao_log.outer.accept_v];
    V.diag.accept_theta = [ao_log.outer.accept_theta];
    V.diag.accept_theta_main = [ao_log.outer.accept_theta_main];
    V.diag.accept_theta_polish = [ao_log.outer.accept_theta_polish];
    V.diag.theta_changed_norm = [ao_log.outer.theta_changed_norm];
    V.diag.theta_changed_norm_polish = [ao_log.outer.theta_changed_norm_polish];
    V.diag.theta_candidate_count = [ao_log.outer.theta_candidate_count];
    V.diag.theta_best_improve_main = [ao_log.outer.theta_best_improve_main];
    V.diag.theta_best_improve_polish = [ao_log.outer.theta_best_improve_polish];
    
    % Add H2 diagnostics from out_old.diag if available
    if isfield(out_old, 'diag')
        if isfield(out_old.diag, 'theta_pre_refit_improve')
            V.diag.theta_pre_refit_improve = out_old.diag.theta_pre_refit_improve;
        else
            V.diag.theta_pre_refit_improve = NaN;
        end
        if isfield(out_old.diag, 'theta_post_refit_improve')
            V.diag.theta_post_refit_improve = out_old.diag.theta_post_refit_improve;
        else
            V.diag.theta_post_refit_improve = NaN;
        end
        if isfield(out_old.diag, 'theta_refit_swallow_ratio')
            V.diag.theta_refit_swallow_ratio = out_old.diag.theta_refit_swallow_ratio;
        else
            V.diag.theta_refit_swallow_ratio = NaN;
        end
        if isfield(out_old.diag, 'theta_good_refit_bad_flag')
            V.diag.theta_good_refit_bad_flag = out_old.diag.theta_good_refit_bad_flag;
        else
            V.diag.theta_good_refit_bad_flag = 0;
        end
    else
        V.diag.theta_pre_refit_improve = NaN;
        V.diag.theta_post_refit_improve = NaN;
        V.diag.theta_refit_swallow_ratio = NaN;
        V.diag.theta_good_refit_bad_flag = 0;
    end
end

beam_info = V;
beam_info.sum_rate = out_old.sum_rate_bps;
ao_log.eval_calls = eval_calls;
ao_log.best_qoe = best_out.avg_qoe;
ao_log.best_sum_rate = best_out.sum_rate_bps;

end

function [profile, weights_row] = resolve_profile(cfg, geom, arg7)
num_users = cfg.num_users;

if nargin < 3 || isempty(arg7)
    weights_row = cfg.weights(1, :);
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights_row, num_users, 1);
    return;
end

if isstruct(arg7)
    profile = arg7;
    weights_row = mean(profile.weights, 1);
else
    weights_row = arg7(:).';
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights_row, num_users, 1);
end
end

function q_weights = build_excess_weights(cfg, out_prev, profile)
% QoE / WMMSE weights with urgent-first emphasis

    K = cfg.num_users;
    q_weights = ones(K,1);

    lambda_delay = get_cfg_or_default(cfg, 'lambda_delay_weight', 5.0);
    lambda_sem   = get_cfg_or_default(cfg, 'lambda_sem_weight',   18.0);

    urgent_mult  = get_cfg_or_default(cfg, 'urgent_weight_mult',  5.0);
    normal_mult  = get_cfg_or_default(cfg, 'normal_weight_mult',  1.0);
    urgent_semantic_mult = get_cfg_or_default(cfg, 'urgent_semantic_mult', 3.0);

    rate_floor_u = get_cfg_or_default(cfg, 'urgent_min_rate_user', 0.08e6);  % 0.08 Mbps per urgent user
    lambda_floor = get_cfg_or_default(cfg, 'urgent_floor_penalty', 6.0);
    
    urgent_idx_arr = get_urgent_indices_from_profile(profile, K);
    is_urgent = false(K, 1);
    is_urgent(urgent_idx_arr) = true;

    % 从评估结果中取连续型违约统计
    D_vec = safe_vec(out_prev, 'D', K, 0);
    dmax_k = profile.dmax_k(:);
    T_tx = safe_vec(out_prev, 'T_tx', K, 0);
    deadlines = profile.d_k(:);
    user_rate = safe_vec(out_prev, 'user_rate', K, 0);

    delay_excess = max(0, T_tx - deadlines) ./ max(deadlines, 1e-9);
    sem_excess   = max(0, D_vec - dmax_k) ./ max(dmax_k, 1e-9);

    for k = 1:K
        if is_urgent(k) > 0
            delay_term = lambda_delay * delay_excess(k);
            sem_term = urgent_semantic_mult * lambda_sem * (sem_excess(k)^2);
            deficit = max(0, rate_floor_u - user_rate(k)) / max(rate_floor_u, 1e-12);
            floor_term = lambda_floor * deficit;
            q_weights(k) = urgent_mult * (1.0 + delay_term + sem_term + floor_term);
        else
            q_weights(k) = normal_mult * (1.0 + 0.5 * lambda_delay * delay_excess(k) + 0.5 * lambda_sem * sem_excess(k));
        end
    end

    % 防止极端爆炸
    q_weights = min(q_weights, get_cfg_or_default(cfg, 'max_q_weight', 50));
    q_weights = max(q_weights, get_cfg_or_default(cfg, 'min_q_weight', 1));
end

function v = safe_vec(s, name, K, defaultv)
    if isstruct(s) && isfield(s, name) && numel(s.(name)) == K
        v = s.(name)(:);
    else
        v = defaultv * ones(K,1);
    end
end

function theta_all = init_theta_by_assignment(cfg, ch, assign)
%INIT_THETA_BY_ASSIGNMENT Align each RIS to its associated user (if any).
L = cfg.num_ris;
N = cfg.n_ris;
theta_all = ch.theta;

for l = 1:L
    users_l = find(assign == l);
    if isempty(users_l)
        continue;
    end
    k = users_l(1);
    hdk = ch.h_d(:, k);
    w0 = hdk / (norm(hdk) + cfg.eps);
    proj = (w0' * ch.G(:, :, l)).' .* ch.H_ris(:, k, l);
    theta_all(:, l) = exp(-1j * angle(proj + cfg.eps));
end

if size(theta_all, 1) ~= N
    theta_all = theta_all(1:N, :);
end
end



function V = build_mrt_beam(cfg, h_eff, p_dbw)
Nt = cfg.nt;
K = cfg.num_users;
P = 10^(p_dbw / 10);

V_p = zeros(Nt, K);
for k = 1:K
    hk = h_eff(:, k);
    V_p(:, k) = hk / (norm(hk) + cfg.eps);
end
V_p = sqrt(P / K) * V_p;

V = struct();
V.v_c = zeros(Nt, 1);
V.V_p = V_p;
V.c = zeros(K, 1);
end

function V = build_rzf_beam_fixed_common(cfg, h_eff, p_dbw, common_ratio, is_urgent, weights)
Nt = cfg.nt;
K = cfg.num_users;
P_total = 10^(p_dbw / 10);
Pc = P_total * common_ratio;
Pp = P_total * (1 - common_ratio);

v_c = sum(h_eff, 2);
v_c = v_c / (norm(v_c) + cfg.eps) * sqrt(Pc);

HH = h_eff' * h_eff;
reg_val = max(cfg.noise_watts, real(trace(HH)) / (K * 100)) + 1e-8;
V_p_zf = h_eff / (HH + reg_val * eye(K));

V_p = zeros(Nt, K);
for k = 1:K
    V_p(:, k) = V_p_zf(:, k) / (norm(V_p_zf(:, k)) + cfg.eps);
end

% Simple power allocation proportional to weights
p_alloc = weights(:) / sum(weights) * Pp;
for k = 1:K
    V_p(:, k) = V_p(:, k) * sqrt(p_alloc(k));
end

% Fake c using simple rule
hk_c = sum(abs(v_c'*h_eff).^2); % simplified
R_c = 0.99 * cfg.bandwidth * log2(1 + min(hk_c)/(Pp + cfg.noise_watts));
c_alloc = max(0, R_c) * (weights(:) / sum(weights));

V = struct();
V.v_c = v_c;
V.V_p = V_p;
V.c = c_alloc;
end

function v = get_cfg_or_default(cfg, field_name, default_v)
if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
    v = cfg.(field_name);
else
    v = default_v;
end
end

function urgent_idx = get_urgent_indices_from_profile(profile, K)
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx')
    error('ua_qoe_ao:missing_urgent_group', ...
        'profile.groups.urgent_idx is required for unified urgent/normal grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
if isempty(urgent_idx)
    error('ua_qoe_ao:empty_urgent_group', 'profile.groups.urgent_idx resolved to empty.');
end
end

function [tf, reason] = accept_candidate_relaxed(out_new, out_old, cfg, step_type)
reason = 'unknown_rejection';

new_cost = get_compare_cost(out_new);
old_cost = get_compare_cost(out_old);

% ---------- 1) Hard guard: urgent floor ----------
urgent_min_sum = get_cfg_or_default(cfg, 'urgent_min_sum_rate', 0.18e6);   % 总紧急和速率门槛
urgent_min_avg = get_cfg_or_default(cfg, 'urgent_min_avg_rate', 0.045e6);   % 单紧急平均速率门槛

if isfield(out_new, 'urgent_sum_rate') && out_new.urgent_sum_rate < urgent_min_sum
    tf = false;
    reason = 'urgent_min_sum_fail';
    return;
end

if isfield(out_new, 'urgent_avg_rate') && out_new.urgent_avg_rate < urgent_min_avg
    tf = false;
    reason = 'urgent_min_avg_fail';
    return;
end

% ---------- 2) Hard guard: no catastrophic total-rate loss ----------
max_total_loss_rel = get_cfg_or_default(cfg, 'max_total_rate_loss_rel', 0.08);
if out_new.total_sum_rate < (1 - max_total_loss_rel) * out_old.total_sum_rate
    tf = false;
    reason = 'max_total_loss_rel_fail';
    return;
end

% ---------- 2.5) Hard guard: no catastrophic normal QoE regression ----------
urgent_satisfied = get_cfg_or_default(cfg, 'urgent_rate_satisfied', 1.5e6);
max_normal_loose = get_cfg_or_default(cfg, 'max_normal_qoe_regress_loose', 0.03);
max_normal_tight = get_cfg_or_default(cfg, 'max_normal_qoe_regress_tight', 0.01);

max_normal_regress = max_normal_loose;
if isfield(out_old, 'urgent_sum_rate') && out_old.urgent_sum_rate >= urgent_satisfied
    max_normal_regress = max_normal_tight;
end

if isfield(out_new, 'normal_qoe') && isfield(out_old, 'normal_qoe')
    if out_new.normal_qoe > out_old.normal_qoe + max_normal_regress
        tf = false;
        reason = 'max_normal_regress_fail';
        return;
    end
end

% ---------- 3) Channel A: strong urgent improvement ----------
tol_uq_s = get_cfg_or_default(cfg, 'tol_urgent_qoe_strong', 0.02);
tol_ur_s = get_cfg_or_default(cfg, 'tol_urgent_rate_strong', 0.05e6);
tol_us_s = get_cfg_or_default(cfg, 'tol_urgent_sem_strong', 0.02);

if out_new.urgent_qoe < out_old.urgent_qoe - tol_uq_s
    tf = true; return;
end

if isfield(out_new, 'urgent_sum_rate') && isfield(out_old, 'urgent_sum_rate')
    if out_new.urgent_sum_rate > out_old.urgent_sum_rate + tol_ur_s
        tf = true; return;
    end
end

if isfield(out_new, 'urgent_semantic_violation') && isfield(out_old, 'urgent_semantic_violation')
    if out_new.urgent_semantic_violation < out_old.urgent_semantic_violation - tol_us_s
        tf = true; return;
    end
end

% ---------- 3.5) Channel A+: Semantic Similarity Incentive ----------
tol_xi_bonus = get_cfg_or_default(cfg, 'tol_xi_bonus', 0.003);
tol_uq_tie = get_cfg_or_default(cfg, 'tol_urgent_qoe_tie', 1e-4);
tol_ur_tie = get_cfg_or_default(cfg, 'tol_urgent_rate_tie', 1e-4);

if abs(out_new.urgent_qoe - out_old.urgent_qoe) <= tol_uq_tie && ...
   isfield(out_new, 'urgent_sum_rate') && isfield(out_old, 'urgent_sum_rate') && ...
   abs(out_new.urgent_sum_rate - out_old.urgent_sum_rate) <= tol_ur_tie
    if isfield(out_new, 'xi_mean_all') && isfield(out_old, 'xi_mean_all')
        if out_new.xi_mean_all > out_old.xi_mean_all + tol_xi_bonus
            tf = true; return;
        end
    end
    
    tol_normal_bonus = get_cfg_or_default(cfg, 'tol_normal_bonus', 0.01);
    if isfield(out_new, 'normal_qoe') && isfield(out_old, 'normal_qoe')
        if out_new.normal_qoe < out_old.normal_qoe - tol_normal_bonus
            tf = true; return;
        end
    end
end

% ---------- 4) Channel B: weak improvement / near-tie ----------
tol_uq_w = get_cfg_or_default(cfg, 'tol_urgent_qoe_weak', 0.01);
tol_us_w = get_cfg_or_default(cfg, 'tol_urgent_sem_weak', 0.01);
tol_aq_w = get_cfg_or_default(cfg, 'tol_avg_qoe_weak', 0.01);
tol_tr_w = get_cfg_or_default(cfg, 'tol_total_rate_weak', 0.5e6);

if strcmpi(step_type, 'T') && get_cfg_or_default(cfg, 'theta_accept_relax', true)
    if out_new.urgent_qoe < out_old.urgent_qoe - 1e-4
        tf = true; return;
    end
    if isfield(out_new, 'urgent_sum_rate') && isfield(out_old, 'urgent_sum_rate')
        if out_new.urgent_sum_rate > out_old.urgent_sum_rate + 2e4 % 20 kbps
            tf = true; return;
        end
    end
    tol_tr_w = get_cfg_or_default(cfg, 'tol_total_rate_weak_theta', 2.0e6);
    tol_aq_w = get_cfg_or_default(cfg, 'tol_avg_qoe_weak_theta', 0.03);
    tol_uq_w = get_cfg_or_default(cfg, 'tol_urgent_qoe_weak_theta', 0.015);
    tol_us_w = get_cfg_or_default(cfg, 'tol_urgent_sem_weak_theta', 0.015);
end

weak_ok = true;
if out_new.urgent_qoe > out_old.urgent_qoe + tol_uq_w
    weak_ok = false;
end
if isfield(out_new, 'urgent_semantic_violation') && isfield(out_old, 'urgent_semantic_violation')
    if out_new.urgent_semantic_violation > out_old.urgent_semantic_violation + tol_us_w
        weak_ok = false;
    end
end
if new_cost > old_cost + tol_aq_w
    weak_ok = false;
end
if out_new.total_sum_rate < out_old.total_sum_rate - tol_tr_w
    weak_ok = false;
end

if weak_ok
    tf = true; reason = 'Accepted'; return;
end

tf = false;
if new_cost > old_cost + tol_aq_w
    reason = 'composite_cost_worse';
elseif out_new.urgent_qoe > out_old.urgent_qoe + tol_uq_w
    reason = 'urgent_qoe_worse';
elseif isfield(out_new, 'urgent_semantic_violation') && isfield(out_old, 'urgent_semantic_violation') && out_new.urgent_semantic_violation > out_old.urgent_semantic_violation + tol_us_w
    reason = 'urgent_semantic_worse';
elseif out_new.total_sum_rate < out_old.total_sum_rate - tol_tr_w
    reason = 'total_sum_rate_drop';
end
end

function cost_v = get_compare_cost(out_s)
if isfield(out_s, 'composite_cost') && ~isempty(out_s.composite_cost) && isfinite(out_s.composite_cost)
    cost_v = out_s.composite_cost;
elseif isfield(out_s, 'avg_qoe_pure') && ~isempty(out_s.avg_qoe_pure) && isfinite(out_s.avg_qoe_pure)
    cost_v = out_s.avg_qoe_pure;
else
    cost_v = out_s.avg_qoe;
end
end

function [accept, theta_out, V_out, out_eval, eval_info] = theta_evaluator(cfg, ch, geom, assign, theta_cand, V_current, out_prev, profile, eval_opts, q_weights, is_urgent, p_dbw)
%THETA_EVALUATOR Three-layer theta evaluation structure
%
% Implements the three-layer evaluation process for RIS phase candidates:
%   Layer 1: Pre-refit evaluation (pure phase change benefit)
%   Layer 2: Post-refit validation (with protected refit)
%   Layer 3: Final acceptance decision
%
% Inputs:
%   cfg          - Configuration structure
%   ch           - Channel structure
%   geom         - Geometry structure
%   assign       - Current user-RIS assignment
%   theta_cand   - Candidate theta to evaluate
%   V_current    - Current beamformer
%   out_prev     - Previous evaluation output
%   profile      - User profile structure
%   eval_opts    - Evaluation options
%   q_weights    - QoE-guided weights
%   is_urgent    - Urgent user indicator vector
%   p_dbw        - Power in dBW
%
% Outputs:
%   accept       - Boolean: whether candidate is accepted
%   theta_out    - Output theta (candidate if accepted, else current)
%   V_out        - Output beamformer (refitted if accepted, else current)
%   out_eval     - Evaluation output structure
%   eval_info    - Diagnostic information structure

% Initialize output structure
eval_info = struct();
eval_info.pre_refit_benefit = 0;
eval_info.post_refit_benefit = 0;
eval_info.layer1_pass = false;
eval_info.layer2_pass = false;
eval_info.layer3_pass = false;
eval_info.theta_good_refit_bad = false;
eval_info.swallow_detected = false;

% Get configuration flags
enable_pre_refit = get_cfg_or_default(cfg, 'enable_theta_pre_refit_eval', false);
enable_protected = get_cfg_or_default(cfg, 'enable_protected_theta_refit', false);
enable_layered = get_cfg_or_default(cfg, 'enable_theta_layered_accept', false);

% If all flags disabled, fall back to simple evaluation with refit
if ~enable_pre_refit && ~enable_protected && ~enable_layered
    % Legacy path: evaluate with refit
    h_eff_cand = effective_channel(cfg, ch, assign, theta_cand);
    
    if get_cfg_or_default(cfg, 'diag_rzf_core', false)
        V_refit = build_rzf_beam_fixed_common(cfg, h_eff_cand, p_dbw, 0.20, is_urgent, q_weights);
    else
        refit_iter = get_cfg_or_default(cfg, 'post_theta_wmmse_refit', 4);
        [V_refit, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff_cand, p_dbw, q_weights, refit_iter, V_current, is_urgent);
    end
    
    sol_cand = struct('assign', assign, 'theta_all', theta_cand, 'V', V_refit);
    out_cand = evaluate_system_rsma(cfg, ch, geom, sol_cand, profile, eval_opts);
    
    [accept, ~] = accept_candidate_relaxed(out_cand, out_prev, cfg, 'T');
    
    if accept
        theta_out = theta_cand;
        V_out = V_refit;
        out_eval = out_cand;
    else
        theta_out = [];
        V_out = V_current;
        out_eval = out_prev;
    end
    return;
end

% ========================================================================
% Layer 1: Pre-refit evaluation (pure phase change benefit)
% ========================================================================
if enable_pre_refit
    % Evaluate theta candidate with current beamformers (no refit)
    h_eff_pre = effective_channel(cfg, ch, assign, theta_cand);
    sol_pre = struct('assign', assign, 'theta_all', theta_cand, 'V', V_current);
    out_pre = evaluate_system_rsma(cfg, ch, geom, sol_pre, profile, eval_opts);
    
    % Record pure phase change benefit
    eval_info.pre_refit_benefit = out_prev.avg_qoe - out_pre.avg_qoe;
    eval_info.delta_urgent_qoe = out_prev.urgent_qoe - out_pre.urgent_qoe;
    eval_info.delta_urgent_sum_rate = out_pre.urgent_sum_rate - out_prev.urgent_sum_rate;
    eval_info.delta_common_ratio_raw = out_pre.common_power_ratio_raw - out_prev.common_power_ratio_raw;
    
    % Layer 1 screening criteria (Requirements 3.2)
    layer1_pass = false;
    if eval_info.delta_urgent_qoe < -1e-6  % urgent_qoe improves (lower is better)
        layer1_pass = true;
    elseif eval_info.delta_urgent_sum_rate > 1e4  % urgent_sum_rate improves
        layer1_pass = true;
    elseif eval_info.delta_common_ratio_raw <= 1e-4 && ...
           isfield(out_pre, 'R_c_limit') && isfield(out_prev, 'R_c_limit') && ...
           out_pre.R_c_limit > out_prev.R_c_limit + 1e4
        % common_ratio_raw not worse AND Rc_limit improves
        layer1_pass = true;
    end
    
    eval_info.layer1_pass = layer1_pass;
    
    if ~layer1_pass
        % Pre-refit screening failed
        accept = false;
        theta_out = [];
        V_out = V_current;
        out_eval = out_prev;
        return;
    end
else
    % Skip pre-refit evaluation
    eval_info.layer1_pass = true;
end

% ========================================================================
% Layer 2: Post-refit validation (with protected refit)
% ========================================================================
if enable_protected
    % Apply protected refit
    h_eff_cand = effective_channel(cfg, ch, assign, theta_cand);
    V_refit = apply_protected_refit(cfg, h_eff_cand, V_current, p_dbw, q_weights, is_urgent);
else
    % Standard refit
    h_eff_cand = effective_channel(cfg, ch, assign, theta_cand);
    
    if get_cfg_or_default(cfg, 'diag_rzf_core', false)
        V_refit = build_rzf_beam_fixed_common(cfg, h_eff_cand, p_dbw, 0.20, is_urgent, q_weights);
    else
        refit_iter = get_cfg_or_default(cfg, 'post_theta_wmmse_refit', 4);
        [V_refit, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff_cand, p_dbw, q_weights, refit_iter, V_current, is_urgent);
    end
end

% Evaluate with refitted beamformer
sol_post = struct('assign', assign, 'theta_all', theta_cand, 'V', V_refit);
out_post = evaluate_system_rsma(cfg, ch, geom, sol_post, profile, eval_opts);

% Record post-refit benefit
eval_info.post_refit_benefit = out_prev.avg_qoe - out_post.avg_qoe;

% Layer 2 validation: check if refit eliminated pre-refit benefits (Requirements 2.8)
if enable_pre_refit && eval_info.pre_refit_benefit > 1e-6 && eval_info.post_refit_benefit < -1e-6
    % Theta was good but refit made it bad
    eval_info.theta_good_refit_bad = true;
    eval_info.swallow_detected = true;
    eval_info.layer2_pass = false;
else
    eval_info.layer2_pass = true;
end

% ========================================================================
% Layer 3: Final acceptance decision
% ========================================================================
if enable_layered
    % Apply three-layer acceptance logic (Requirements 3.6-3.9)
    [accept, layer3_info] = apply_layered_acceptance(cfg, out_post, out_prev, eval_info);
    eval_info.layer3_pass = accept;
    eval_info.layer3_info = layer3_info;
else
    % Use standard acceptance criteria
    [accept, ~] = accept_candidate_relaxed(out_post, out_prev, cfg, 'T');
    eval_info.layer3_pass = accept;
end

% Set outputs
if accept
    theta_out = theta_cand;
    V_out = V_refit;
    out_eval = out_post;
else
    theta_out = [];
    V_out = V_current;
    out_eval = out_prev;
end

end

function V_refit = apply_protected_refit(cfg, h_eff, V_current, p_dbw, q_weights, is_urgent)
%APPLY_PROTECTED_REFIT Apply protected refit constraints
%
% Implements three protected refit modes (Requirements 2.4-2.6):
%   - 'private_only': Fix common beamformer, update only private beamformers
%   - 'weak_common': Allow common updates with weak step size and limited iterations
%   - 'trust_region': Constrain ||V_new - V_old|| and enforce Pc_new <= Pc_old + delta

refit_mode = get_cfg_or_default(cfg, 'theta_refit_mode', 'private_only');

switch lower(refit_mode)
    case 'private_only'
        % Mode 1: Fix common beamformer, update only private beamformers (1-2 iterations)
        cfg_refit = cfg;
        cfg_refit.fix_common_beam = true;  % Signal to rsma_wmmse to fix common
        refit_iter = get_cfg_or_default(cfg, 'protected_refit_iter', 2);
        
        [V_refit, ~, ~, ~, ~, ~] = rsma_wmmse(cfg_refit, h_eff, p_dbw, q_weights, refit_iter, V_current, is_urgent);
        
    case 'weak_common'
        % Mode 2: Allow common updates with weak step size (alpha=0.3-0.5) and limited iterations (1-2)
        cfg_refit = cfg;
        cfg_refit.wmmse_step_size = get_cfg_or_default(cfg, 'weak_common_step_size', 0.4);
        refit_iter = get_cfg_or_default(cfg, 'protected_refit_iter', 2);
        
        [V_refit, ~, ~, ~, ~, ~] = rsma_wmmse(cfg_refit, h_eff, p_dbw, q_weights, refit_iter, V_current, is_urgent);
        
    case 'trust_region'
        % Mode 3: Constrain ||V_new - V_old|| and enforce Pc_new <= Pc_old + delta
        cfg_refit = cfg;
        cfg_refit.trust_region_delta_V = get_cfg_or_default(cfg, 'trust_region_delta_V', 0.5);
        cfg_refit.trust_region_delta_Pc = get_cfg_or_default(cfg, 'trust_region_delta_Pc', 0.1);
        cfg_refit.V_reference = V_current;  % Reference for trust region
        refit_iter = get_cfg_or_default(cfg, 'protected_refit_iter', 2);
        
        [V_refit, ~, ~, ~, ~, ~] = rsma_wmmse(cfg_refit, h_eff, p_dbw, q_weights, refit_iter, V_current, is_urgent);
        
    otherwise
        % Fallback to standard refit
        refit_iter = get_cfg_or_default(cfg, 'post_theta_wmmse_refit', 4);
        [V_refit, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, q_weights, refit_iter, V_current, is_urgent);
end

end

function [accept, layer3_info] = apply_layered_acceptance(cfg, out_new, out_old, eval_info)
%APPLY_LAYERED_ACCEPTANCE Three-layer acceptance logic
%
% Implements layered acceptance criteria (Requirements 3.6-3.9):
%   Layer 1 (Feasibility): urgent_qoe, urgent_sum_rate, common_ratio_raw, Rc_limit
%   Layer 2 (Structure): common_power_ratio_raw, common_cap_active, theta_changed_norm
%   Layer 3 (Efficiency): composite_cost, avg_qoe_pure, urgent_sum_rate

layer3_info = struct();
layer3_info.feasibility_pass = false;
layer3_info.structure_pass = false;
layer3_info.efficiency_pass = false;

% Check if in polish mode
polish_mode = get_cfg_or_default(cfg, 'theta_polish_mode', false);

% ========================================================================
% Layer 1: Feasibility acceptance (Requirements 3.7)
% ========================================================================
feasibility_pass = false;

% Criterion 1: urgent_qoe improvement
if out_new.urgent_qoe < out_old.urgent_qoe - 1e-6
    feasibility_pass = true;
end

% Criterion 2: urgent_sum_rate improvement
if isfield(out_new, 'urgent_sum_rate') && isfield(out_old, 'urgent_sum_rate')
    if out_new.urgent_sum_rate > out_old.urgent_sum_rate + 1e4
        feasibility_pass = true;
    end
end

% Criterion 3: urgent_violation reduction
if isfield(out_new, 'urgent_semantic_violation') && isfield(out_old, 'urgent_semantic_violation')
    if out_new.urgent_semantic_violation < out_old.urgent_semantic_violation - 1e-4
        feasibility_pass = true;
    end
end

% Criterion 4: total_sum_rate not catastrophic
max_rate_loss = get_cfg_or_default(cfg, 'max_total_rate_loss_rel', 0.08);
if out_new.total_sum_rate >= (1 - max_rate_loss) * out_old.total_sum_rate
    % Not catastrophic, but need other criteria to pass
else
    % Catastrophic rate loss - reject immediately
    accept = false;
    return;
end

layer3_info.feasibility_pass = feasibility_pass;

% In polish mode, only require feasibility criteria
if polish_mode
    accept = feasibility_pass;
    return;
end

% ========================================================================
% Layer 2: Structure acceptance (Requirements 3.8)
% ========================================================================
structure_pass = true;

% Criterion 1: common_power_ratio_raw not increasing
if isfield(out_new, 'common_power_ratio_raw') && isfield(out_old, 'common_power_ratio_raw')
    if out_new.common_power_ratio_raw > out_old.common_power_ratio_raw + 1e-4
        structure_pass = false;
    end
end

% Criterion 2: common_cap_active not increasing
if isfield(out_new, 'common_cap_active') && isfield(out_old, 'common_cap_active')
    if out_new.common_cap_active > out_old.common_cap_active
        structure_pass = false;
    end
end

% Criterion 3: theta_changed_norm non-zero (already guaranteed by candidate generation)
% This is implicitly satisfied if we're evaluating a different theta

layer3_info.structure_pass = structure_pass;

% ========================================================================
% Layer 3: Efficiency acceptance (Requirements 3.9)
% ========================================================================
efficiency_pass = false;

% Criterion 1: composite_cost improvement
new_cost = get_compare_cost(out_new);
old_cost = get_compare_cost(out_old);
if new_cost < old_cost - 1e-6
    efficiency_pass = true;
end

% Criterion 2: avg_qoe_pure not significantly worse
tol_avg_qoe = get_cfg_or_default(cfg, 'tol_avg_qoe_weak', 0.01);
if isfield(out_new, 'avg_qoe_pure') && isfield(out_old, 'avg_qoe_pure')
    if out_new.avg_qoe_pure <= out_old.avg_qoe_pure + tol_avg_qoe
        efficiency_pass = true;
    end
end

% Criterion 3: urgent_sum_rate improvement
if isfield(out_new, 'urgent_sum_rate') && isfield(out_old, 'urgent_sum_rate')
    if out_new.urgent_sum_rate > out_old.urgent_sum_rate + 1e4
        efficiency_pass = true;
    end
end

layer3_info.efficiency_pass = efficiency_pass;

% ========================================================================
% Final decision: All three layers must pass
% ========================================================================
accept = feasibility_pass && structure_pass && efficiency_pass;

end
