function [assign, theta_all, beam_info, ao_log] = ua_qoe_ao(cfg, ch, geom, p_dbw, semantic_mode, table_path, weights_or_profile)
%UA_QOE_AO UA-QoE-AO with unified RSMA evaluation.
%
% 7th arg compatibility:
%   - 1x2 row vector: legacy weights row
%   - struct: profile with per-user fields (M_k, weights, d_k, dmax_k, ...)

num_users = cfg.num_users;
num_ris = cfg.num_ris;
n_ris = cfg.n_ris;

[profile, weights_row] = resolve_profile(cfg, geom, weights_or_profile);

% Step 1: initialize assignment (warm-start if provided; else QoE-gain matching)
if isfield(cfg, 'ao_init_assign') && ~isempty(cfg.ao_init_assign) && numel(cfg.ao_init_assign) == num_users
    assign = cfg.ao_init_assign(:);
    assign = max(0, min(num_ris, round(assign)));
else
    assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights_row, geom, profile);
end

% Step 2: initialize theta (align each RIS to its currently-associated user if any)
theta_all = init_theta_by_assignment(cfg, ch, assign);

% Step 3: initialize RSMA precoder
h_eff = effective_channel(cfg, ch, assign, theta_all);
simple_beam = get_cfg_or_default(cfg, 'ao_simple_beam', false);
if simple_beam
    V = build_mrt_beam(cfg, h_eff, p_dbw);
else
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(num_users, 1), 2);
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

eval_opts = struct('semantic_mode', semantic_mode, 'table_path', table_path);
eval_calls = 0;
urgent_idx = get_urgent_indices_from_profile(profile, num_users);

sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
out_old = evaluate_system_rsma(cfg, ch, geom, sol, profile, eval_opts);
eval_calls = eval_calls + 1;
ao_log.qoe_history = out_old.avg_qoe;
ao_log.sum_rate_history = out_old.sum_rate_bps;
ao_log.accept.assign = 0;
ao_log.accept.V = 0;
ao_log.accept.theta = 0;
ao_log.accept.rand = 0;
ao_log.outer = struct('accept_assign', {}, 'accept_v', {}, 'accept_theta', {}, 'accept_rand', {}, ...
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
        sol_try_a = struct('assign', assign_cand, 'theta_all', theta_all, 'V', V);
        out_try_a = evaluate_system_rsma(cfg, ch, geom, sol_try_a, profile, eval_opts);
        eval_calls = eval_calls + 1;
        margin = max(accept_abs, accept_rel * abs(out_prev.avg_qoe));
        if out_try_a.avg_qoe <= out_prev.avg_qoe + margin
            assign = assign_cand;
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
        [V_cand, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, q_weights, wmmse_iter);
        sol_cand_v = struct('assign', assign, 'theta_all', theta_all, 'V', V_cand);
        out_cand_v = evaluate_system_rsma(cfg, ch, geom, sol_cand_v, profile, eval_opts);
        eval_calls = eval_calls + 1;
        if verbose
            fprintf('[ua][outer %d][V] cand cost=%.6f\n', outer, out_cand_v.avg_qoe);
        end

        alpha_v = 1.0;
        for t = 1:bt_max_tries
            V_try = blend_struct(V_prev, V_cand, alpha_v);
            sol_try_v = struct('assign', assign, 'theta_all', theta_all, 'V', V_try);
            out_try_v = evaluate_system_rsma(cfg, ch, geom, sol_try_v, profile, eval_opts);
            eval_calls = eval_calls + 1;
            if verbose && t <= 3
                fprintf('[ua][outer %d][V] try alpha=%.3f cost=%.6f\n', outer, alpha_v, out_try_v.avg_qoe);
            end

            margin = max(accept_abs, accept_rel * abs(out_prev.avg_qoe));
            if out_try_v.avg_qoe <= out_prev.avg_qoe + margin
                V = V_try;
                out_prev = out_try_v;
                ao_log.qoe_history(end + 1, 1) = out_prev.avg_qoe;
                ao_log.sum_rate_history(end + 1, 1) = out_prev.sum_rate_bps;
                ao_log.accept.V = ao_log.accept.V + 1;
                if verbose
                    fprintf('[ua][outer %d][V] accept alpha=%.3f cost=%.6f\n', outer, alpha_v, out_prev.avg_qoe);
                end
                accepted_v = true;
                break;
            end

            alpha_v = alpha_v * 0.5;
        end
    end

    if ~accepted_v
        V = V_prev;
        if verbose
            fprintf('[ua][outer %d][V] no-accept\n', outer);
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
        theta_cand = ris_phase_mm(cfg, ch, assign, V, q_weights, mm_iter, theta_all);
        sol_cand_t = struct('assign', assign, 'theta_all', theta_cand, 'V', V);
        out_cand_t = evaluate_system_rsma(cfg, ch, geom, sol_cand_t, profile, eval_opts);
        eval_calls = eval_calls + 1;
        if verbose
            fprintf('[ua][outer %d][T] cand cost=%.6f\n', outer, out_cand_t.avg_qoe);
        end

        alpha_t = 1.0;
        for t = 1:bt_max_tries
            theta_try = exp(1j * angle((1 - alpha_t) * theta_prev + alpha_t * theta_cand));
            sol_try_t = struct('assign', assign, 'theta_all', theta_try, 'V', V);
            out_try_t = evaluate_system_rsma(cfg, ch, geom, sol_try_t, profile, eval_opts);
            eval_calls = eval_calls + 1;
            if verbose && t <= 3
                fprintf('[ua][outer %d][T] try alpha=%.3f cost=%.6f\n', outer, alpha_t, out_try_t.avg_qoe);
            end

            margin = max(accept_abs, accept_rel * abs(out_prev.avg_qoe));
            if out_try_t.avg_qoe <= out_prev.avg_qoe + margin
                theta_all = theta_try;
                out_prev = out_try_t;
                ao_log.qoe_history(end + 1, 1) = out_prev.avg_qoe;
                ao_log.sum_rate_history(end + 1, 1) = out_prev.sum_rate_bps;
                ao_log.accept.theta = ao_log.accept.theta + 1;
                if verbose
                    fprintf('[ua][outer %d][T] accept alpha=%.3f cost=%.6f\n', outer, alpha_t, out_prev.avg_qoe);
                end
                accepted_t = true;
                break;
            end

            alpha_t = alpha_t * 0.5;
        end

        if ~accepted_t && verbose
            fprintf('[ua][outer %d][T] no-accept\n', outer);
        end
    end

    % Step 4: random perturbation fallback when both V-step and T-step fail
    if ~freeze_theta && ~disable_random_fallback && ~accepted_v && ~accepted_t
        best_cost = inf;
        best_theta = theta_prev;
        best_out_r = out_prev;
        ris_used = unique(assign(assign > 0)).';
        if isempty(ris_used)
            ris_used = 1:num_ris;
        end

        for r = 1:rand_tries
            theta_try = theta_prev;
            for l = ris_used
                ph = angle(theta_prev(:, l)) + rand_sigma * randn(size(theta_prev(:, l)));
                theta_try(:, l) = exp(1j * ph);
            end
            sol_try_r = struct('assign', assign, 'theta_all', theta_try, 'V', V_prev);
            out_try_r = evaluate_system_rsma(cfg, ch, geom, sol_try_r, profile, eval_opts);
            eval_calls = eval_calls + 1;
            if out_try_r.avg_qoe < best_cost
                best_cost = out_try_r.avg_qoe;
                best_theta = theta_try;
                best_out_r = out_try_r;
            end
        end

        margin = max(accept_abs, accept_rel * abs(out_prev.avg_qoe));
        if best_cost <= out_prev.avg_qoe + margin
            theta_all = best_theta;
            V = V_prev;
            out_prev = best_out_r;
            ao_log.qoe_history(end + 1, 1) = out_prev.avg_qoe;
            ao_log.sum_rate_history(end + 1, 1) = out_prev.sum_rate_bps;
            ao_log.accept.rand = ao_log.accept.rand + 1;
            accepted_r = true;
            if verbose
                fprintf('[ua][outer %d][R] accept cost=%.6f\n', outer, out_prev.avg_qoe);
            end
        else
            if verbose
                fprintf('[ua][outer %d][R] no-accept\n', outer);
            end
        end
    end

    if out_prev.avg_qoe < best_out.avg_qoe - tol
        best_out = out_prev;
        best_sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
    end

    outer_log = struct();
    outer_log.accept_assign = accepted_assign;
    outer_log.accept_v = accepted_v;
    outer_log.accept_theta = accepted_t;
    outer_log.accept_rand = accepted_r;
    outer_log.avg_qoe = out_prev.avg_qoe;
    outer_log.urgent_qoe = mean(out_prev.qoe_vec(urgent_idx));
    outer_log.sum_rate = out_prev.sum_rate_bps;
    outer_log.break_reason = '';
    ao_log.outer(end + 1, 1) = outer_log;
    out_old = out_prev;

    % Early stop if nothing was accepted in this outer iteration.
    if outer >= min_outer_iter && ~accepted_assign && ~accepted_v && ~accepted_t && ~accepted_r
        ao_log.outer(end).break_reason = 'no_accept';
        if verbose
            fprintf('[ua][outer %d] no improvement -> stop\n', outer);
        end
        break;
    end
end

assign = best_sol.assign;
theta_all = best_sol.theta_all;
V = best_sol.V;
out_old = best_out;

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

function q_weights = build_excess_weights(cfg, out_old, profile)
K = cfg.num_users;
ratio_d = out_old.T_tx ./ (profile.d_k(:) + cfg.eps);
ratio_s = out_old.D ./ (profile.dmax_k(:) + cfg.eps);

% Keep weight construction consistent with QoE evaluation (ratio is clipped in qoe.m)
if isfield(cfg, 'max_ratio_d')
    ratio_d = min(ratio_d, cfg.max_ratio_d);
end
if isfield(cfg, 'max_ratio_s')
    ratio_s = min(ratio_s, cfg.max_ratio_s);
end

% Violation excess (>=0)
ex_d = max(0, ratio_d - 1);
ex_s = max(0, ratio_s - 1);

% Per-user delay/semantic importance
w_d = profile.weights(:, 1);
w_s = profile.weights(:, 2);

% Fairness floor + weighted excess emphasis
q_weights = 1 + w_d .* ex_d + w_s .* ex_s;

% Normalize by mean to keep weights O(1)
m = mean(q_weights);
if ~isfinite(m) || m <= cfg.eps || any(~isfinite(q_weights))
    q_weights = ones(K, 1);
else
    q_weights = q_weights / (m + cfg.eps);
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

function V_out = blend_struct(V_old, V_cand, alpha)
V_out = V_old;
V_out.v_c = (1 - alpha) * V_old.v_c + alpha * V_cand.v_c;
V_out.V_p = (1 - alpha) * V_old.V_p + alpha * V_cand.V_p;
V_out.c = (1 - alpha) * V_old.c + alpha * V_cand.c;
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
