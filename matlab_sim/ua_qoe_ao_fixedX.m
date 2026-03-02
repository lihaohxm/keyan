function [sol, log] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign_fixed, profile, opts)
%UA_QOE_AO_FIXEDX AO with fixed assignment X: optimize RSMA + RIS phase only.

if nargin < 7 || isempty(opts), opts = struct(); end
if nargin < 6, profile = []; end

K = cfg.num_users;
L = cfg.num_ris;
N = cfg.n_ris;

assign_fixed = assign_fixed(:);
if numel(assign_fixed) ~= K
    error('assign_fixed must be Kx1.');
end

max_outer_iter = get_opt(opts, 'max_outer_iter', 5);
wmmse_iter = get_opt(opts, 'wmmse_iter', 5);
mm_iter = get_opt(opts, 'mm_iter', get_opt(cfg, 'ris_mm_iter', 3));
bt_max_tries = get_opt(opts, 'bt_max_tries', 6);
bt_verbose = get_opt(opts, 'bt_verbose', false);
init_theta = get_opt(opts, 'init_theta', 'align');
verbose = get_opt(opts, 'verbose', false);
if verbose
    fprintf('[fixedX] max_outer_iter=%d\n', max_outer_iter);
end

% Build evaluation opts
opts_eval = struct();
if isfield(opts, 'semantic_mode'), opts_eval.semantic_mode = opts.semantic_mode; end
if isfield(opts, 'table_path'), opts_eval.table_path = opts.table_path; end
if isfield(opts, 'semantic_params'), opts_eval.semantic_params = opts.semantic_params; end
if isfield(opts, 'prop_delay_opts'), opts_eval.prop_delay_opts = opts.prop_delay_opts; end

% 1) theta_all init
if strcmpi(init_theta, 'random')
    theta_all = ch.theta;
else
    theta_all = init_theta_by_assignment(cfg, ch, assign_fixed);
end

% 2) V init
h_eff = effective_channel(cfg, ch, assign_fixed, theta_all);
[V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(K, 1), 2);

sol = struct();
sol.assign = assign_fixed;
sol.theta_all = theta_all;
sol.V = V;

out_old = evaluate_system_rsma(cfg, ch, geom, sol, profile, opts_eval);
log.qoe_history = out_old.avg_qoe;
log.sum_rate_history = out_old.sum_rate_bps;
tol = 1e-9;

% 3) AO outer loop
for outer = 1:max_outer_iter
    V_old = sol.V;
    theta_old = sol.theta_all;

    ratio_d = out_old.T_tx ./ (profile.d_k(:) + cfg.eps);
    ratio_s = out_old.D ./ (profile.dmax_k(:) + cfg.eps);
    if isfield(cfg, 'max_ratio_d')
        ratio_d = min(ratio_d, cfg.max_ratio_d);
    end
    if isfield(cfg, 'max_ratio_s')
        ratio_s = min(ratio_s, cfg.max_ratio_s);
    end
    ex_d = max(0, ratio_d - 1);
    ex_s = max(0, ratio_s - 1);

    w_d = profile.weights(:, 1);
    w_s = profile.weights(:, 2);
    q_weights = 1 + w_d .* ex_d + w_s .* ex_s;

    m = mean(q_weights);
    if ~isfinite(m) || m <= cfg.eps || any(~isfinite(q_weights))
        q_weights = ones(K, 1);
    else
        q_weights = q_weights / (m + cfg.eps);
    end

    % b) RSMA update
    h_eff = effective_channel(cfg, ch, assign_fixed, sol.theta_all);
    [V_cand, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, q_weights, wmmse_iter);

    % c) RIS phase update
    theta_cand = ris_phase_mm(cfg, ch, assign_fixed, V_cand, q_weights, mm_iter, sol.theta_all);

    % Candidate quality at alpha = 1 (for observability)
    sol_new = sol;
    sol_new.V = V_cand;
    sol_new.theta_all = theta_cand;
    out_new = evaluate_system_rsma(cfg, ch, geom, sol_new, profile, opts_eval);
    if verbose
        fprintf('[fixedX][outer %d] old=%.6f new=%.6f (tol=%.3e)\n', ...
            outer, out_old.avg_qoe, out_new.avg_qoe, tol);
    end

    % d) Backtracking with damping/line-search
    accepted = false;
    alpha = 1.0;
    for bt = 1:bt_max_tries
        V_try = blend_struct(V_old, V_cand, alpha);
        theta_try = exp(1j * angle((1 - alpha) * theta_old + alpha * theta_cand));

        sol_try = sol;
        sol_try.V = V_try;
        sol_try.theta_all = theta_try;
        out_try = evaluate_system_rsma(cfg, ch, geom, sol_try, profile, opts_eval);

        if verbose && bt_verbose
            fprintf('[fixedX][outer %d][bt %d] alpha=%.3f cost=%.6f\n', ...
                outer, bt, alpha, out_try.avg_qoe);
        end

        if out_try.avg_qoe <= out_old.avg_qoe + tol
            sol = sol_try;
            out_old = out_try;
            log.qoe_history(end+1, 1) = out_old.avg_qoe;
            log.sum_rate_history(end+1, 1) = out_old.sum_rate_bps;
            if verbose
                fprintf('[fixedX][outer %d] accept alpha=%.3f cost=%.6f\n', ...
                    outer, alpha, out_old.avg_qoe);
            end
            accepted = true;
            break;
        end

        alpha = alpha * 0.5;
    end

    if ~accepted
        sol.V = V_old;
        sol.theta_all = theta_old;
        continue;
    end
end

end

function V_out = blend_struct(V_old, V_cand, alpha)
V_out = V_old;
V_out.v_c = (1 - alpha) * V_old.v_c + alpha * V_cand.v_c;
V_out.V_p = (1 - alpha) * V_old.V_p + alpha * V_cand.V_p;
V_out.c = (1 - alpha) * V_old.c + alpha * V_cand.c;
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

function v = get_opt(opts, f, d)
if isfield(opts, f) && ~isempty(opts.(f))
    v = opts.(f);
else
    v = d;
end
end
