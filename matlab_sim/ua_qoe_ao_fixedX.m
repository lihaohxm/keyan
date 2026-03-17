function [sol, log] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign_fixed, profile, opts)
%UA_QOE_AO_FIXEDX AO with fixed assignment X: optimize RSMA + RIS phase only.

if nargin < 7 || isempty(opts), opts = struct(); end
if nargin < 6, profile = []; end

K = cfg.num_users;

if isempty(profile)
    profile = build_profile_urgent_normal(cfg, geom, struct());
end
cfg = attach_profile_context_to_cfg(cfg, profile);

assign_fixed = assign_fixed(:);
if numel(assign_fixed) ~= K
    error('assign_fixed must be Kx1.');
end

max_outer_iter = get_opt(opts, 'max_outer_iter', 5);
wmmse_iter = get_opt(opts, 'wmmse_iter', 5);
mm_iter = get_opt(opts, 'mm_iter', get_opt(cfg, 'ris_mm_iter', 3));
init_theta = get_opt(opts, 'init_theta', 'align');
verbose = get_opt(opts, 'verbose', false);
if verbose
    fprintf('[fixedX] max_outer_iter=%d\n', max_outer_iter);
end

opts_eval = struct();
if isfield(opts, 'semantic_mode'), opts_eval.semantic_mode = opts.semantic_mode; end
if isfield(opts, 'table_path'), opts_eval.table_path = opts.table_path; end
if isfield(opts, 'semantic_params'), opts_eval.semantic_params = opts.semantic_params; end
if isfield(opts, 'prop_delay_opts'), opts_eval.prop_delay_opts = opts.prop_delay_opts; end

if strcmpi(init_theta, 'random')
    theta_all = ch.theta;
else
    theta_all = init_theta_by_assignment(cfg, ch, assign_fixed);
end

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

for outer = 1:max_outer_iter
    V_old = sol.V;
    theta_old = sol.theta_all;

    is_baseline = get_opt(opts, 'baseline_mode', false);
    q_weights = build_task_priority_weights(cfg, out_old, profile, ...
        struct('baseline_mode', is_baseline, 'use_gradient_term', true));

    h_eff = effective_channel(cfg, ch, assign_fixed, sol.theta_all);
    cfg_wmmse = cfg;
    cfg_wmmse.urgent_delay_vio = get_metric_or_default(out_old, 'urgent_delay_violation', 1.0);
    cfg_wmmse.urgent_sem_vio = get_metric_or_default(out_old, 'urgent_semantic_violation', 1.0);
    [V_cand, ~, ~, ~, ~, ~] = rsma_wmmse(cfg_wmmse, h_eff, p_dbw, q_weights, wmmse_iter);

    theta_cand = ris_phase_mm(cfg, ch, assign_fixed, V_cand, q_weights, mm_iter, sol.theta_all);

    sol_new = sol;
    sol_new.V = V_cand;
    sol_new.theta_all = theta_cand;
    out_new = evaluate_system_rsma(cfg, ch, geom, sol_new, profile, opts_eval);
    if verbose
        fprintf('[fixedX][outer %d] old=%.6f new=%.6f (tol=%.3e)\n', ...
            outer, out_old.avg_qoe, out_new.avg_qoe, tol);
    end

    if accept_fixedx_candidate(out_new, out_old, tol)
        sol.V = V_cand;
        sol.theta_all = theta_cand;
        out_old = out_new;

        log.qoe_history(end + 1, 1) = out_old.avg_qoe;
        log.sum_rate_history(end + 1, 1) = out_old.sum_rate_bps;

        if verbose
            fprintf('[fixedX][outer %d] accept cost=%.6f\n', outer, out_old.avg_qoe);
        end
    else
        sol.V = V_old;
        sol.theta_all = theta_old;
    end
end

end

function tf = accept_fixedx_candidate(out_new, out_old, tol)
new_cost = get_compare_cost(out_new);
old_cost = get_compare_cost(out_old);
tf = (new_cost <= old_cost + tol);
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

function value = get_metric_or_default(s, name, default_value)
if isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
end
