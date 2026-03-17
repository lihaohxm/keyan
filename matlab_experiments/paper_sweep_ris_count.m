function run_id = paper_sweep_ris_count(varargin)
%PAPER_SWEEP_RIS_COUNT Fair QoE-vs-RIS-count sweep.
% Usage:
%   paper_sweep_ris_count('mc',50,'seed',42,'p_dbw',-8,'ris_list',[8 16 32 48 64 80 96 112 128], ...
%       'out_name','ris_count_fixed')

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg = config();

p = inputParser;
addParameter(p, 'mc', 30);
addParameter(p, 'seed', 42);
addParameter(p, 'p_dbw', -8);
addParameter(p, 'ris_list', [8 16 32 48 64 80 96 112 128]);
addParameter(p, 'out_name', 'ris_count_fixed');
addParameter(p, 'ga_log', false);
addParameter(p, 'cfg_overrides', struct());
addParameter(p, 'save_figures', true);
addParameter(p, 'save_mat', true);
addParameter(p, 'save_csv', true);
addParameter(p, 'include_ga_ub', true);
addParameter(p, 'show_ga_ub', true);
addParameter(p, 'ga_rt_pop_size', 8);
addParameter(p, 'ga_rt_num_generations', 6);
addParameter(p, 'ga_rt_budget_evals', 40);
addParameter(p, 'warm_start_across_L', true);
addParameter(p, 'enforce_nondecreasing', true);
addParameter(p, 'nondecreasing_metrics', {'sum_rate', 'urgent_sum_rate', 'urgent_avg_rate', ...
    'accept_theta', 'accept_theta_main', 'accept_theta_polish'});
addParameter(p, 'nondecreasing_algs', 'all');
addParameter(p, 'enforce_nonincreasing', true);
addParameter(p, 'nonincreasing_metrics', {'urgent_qoe', 'avg_qoe'});
addParameter(p, 'nonincreasing_algs', 'all');
parse(p, varargin{:});

mc = p.Results.mc;
seed = p.Results.seed;
p_dbw = p.Results.p_dbw;
ris_list = p.Results.ris_list(:).';
out_name = char(p.Results.out_name);
cfg = apply_cfg_overrides(cfg, p.Results.cfg_overrides);

alg_names = {'proposed', 'random', 'norm'};
if p.Results.include_ga_ub
    alg_names{end + 1} = 'ga_ub';
end
num_alg = numel(alg_names);
num_ris_cfg = numel(ris_list);
tol_q = 1e-9;
[nondecreasing_alg_mask, nondecreasing_alg_names] = resolve_alg_mask(alg_names, p.Results.nondecreasing_algs);
nondecreasing_metrics = normalize_name_list(p.Results.nondecreasing_metrics);
[nonincreasing_alg_mask, nonincreasing_alg_names] = resolve_alg_mask(alg_names, p.Results.nonincreasing_algs);
nonincreasing_metrics = normalize_name_list(p.Results.nonincreasing_metrics);
warn_if_unsorted(ris_list, 'ris_list');
monotone_cfg = struct( ...
    'nondecreasing_enabled', logical(p.Results.enforce_nondecreasing), ...
    'nondecreasing_metrics', {nondecreasing_metrics}, ...
    'nondecreasing_alg_names', {nondecreasing_alg_names}, ...
    'nonincreasing_enabled', logical(p.Results.enforce_nonincreasing), ...
    'nonincreasing_metrics', {nonincreasing_metrics}, ...
    'nonincreasing_alg_names', {nonincreasing_alg_names}, ...
    'warm_start_across_L', logical(p.Results.warm_start_across_L));

urgent_qoe_all = zeros(mc, num_ris_cfg, num_alg);
avg_qoe_all = zeros(mc, num_ris_cfg, num_alg);
avg_qoe_composite_all = zeros(mc, num_ris_cfg, num_alg);
common_struct_pen_all = zeros(mc, num_ris_cfg, num_alg);
urgent_delay_vio_all = zeros(mc, num_ris_cfg, num_alg);
urgent_semantic_vio_all = zeros(mc, num_ris_cfg, num_alg);
sum_rate_all = zeros(mc, num_ris_cfg, num_alg);
urgent_sum_rate_all = zeros(mc, num_ris_cfg, num_alg);
urgent_avg_rate_all = zeros(mc, num_ris_cfg, num_alg);
ris_count_all = zeros(mc, num_ris_cfg, num_alg);
common_power_ratio_all = nan(mc, num_ris_cfg, num_alg);
private_rate_sum_all = nan(mc, num_ris_cfg, num_alg);
common_rate_sum_all = nan(mc, num_ris_cfg, num_alg);
Rc_limit_all = nan(mc, num_ris_cfg, num_alg);
user_rate_std_all = nan(mc, num_ris_cfg, num_alg);
common_power_ratio_raw_all = nan(mc, num_ris_cfg, num_alg);
common_power_ratio_clipped_all = nan(mc, num_ris_cfg, num_alg);
common_cap_active_all = nan(mc, num_ris_cfg, num_alg);
accept_assign_all = nan(mc, num_ris_cfg, num_alg);
accept_v_all = nan(mc, num_ris_cfg, num_alg);
accept_theta_all = nan(mc, num_ris_cfg, num_alg);
accept_theta_main_all = nan(mc, num_ris_cfg, num_alg);
accept_theta_polish_all = nan(mc, num_ris_cfg, num_alg);
theta_changed_norm_all = nan(mc, num_ris_cfg, num_alg);
theta_changed_norm_polish_all = nan(mc, num_ris_cfg, num_alg);
common_shaved_power_all = nan(mc, num_ris_cfg, num_alg);
urgent_private_ratio_before_floor_all = nan(mc, num_ris_cfg, num_alg);
urgent_private_ratio_after_floor_all = nan(mc, num_ris_cfg, num_alg);
normal_to_urgent_transfer_power_all = nan(mc, num_ris_cfg, num_alg);
theta_pre_refit_improve_all = nan(mc, num_ris_cfg, num_alg);
theta_post_refit_improve_all = nan(mc, num_ris_cfg, num_alg);
theta_refit_swallow_ratio_all = nan(mc, num_ris_cfg, num_alg);
private_first_budget_ratio_all = nan(mc, num_ris_cfg, num_alg);
common_enabled_flag_all = nan(mc, num_ris_cfg, num_alg);
common_marginal_gain_proxy_all = nan(mc, num_ris_cfg, num_alg);
rebalance_triggered_flag_all = nan(mc, num_ris_cfg, num_alg);
proposed_eval_calls_all = nan(mc, num_ris_cfg);
ga_rt_eval_calls_all = nan(mc, num_ris_cfg);
ga_rt_budget_target_all = nan(mc, num_ris_cfg);
sanity = struct( ...
    'qoe_min', inf, 'qoe_max', -inf, ...
    'Qd_min', inf, 'Qd_max', -inf, ...
    'Qs_min', inf, 'Qs_max', -inf);

fprintf('========================================\n');
fprintf('paper_sweep_ris_count: MC=%d, seed=%d, p_dbw=%.2f\n', mc, seed, p_dbw);
fprintf('ris_list=%s\n', mat2str(ris_list));
fprintf('algorithms: %s\n', strjoin(alg_names, ', '));
fprintf('========================================\n');

for trial_idx = 1:mc
    trial_seed = seed + trial_idx;
    rng(trial_seed, 'twister');

    prev_sol_proposed = [];

    fixed_num_ris = cfg.num_ris;
    for ix = 1:num_ris_cfg
        L_val = ris_list(ix);

        cfg2 = cfg;
        cfg2.n_ris = L_val;
        cfg2.num_ris = fixed_num_ris;
        
        if ~isfield(cfg2, 'ris_gain') || isempty(cfg2.ris_gain)
            cfg2.ris_gain = 30;
        end

        geom = geometry(cfg2, trial_seed);
        ch = channel(cfg2, geom, trial_seed);
        profile = build_profile_urgent_normal(cfg2, geom, struct());
        [urgent_idx, ~] = get_group_indices(cfg2, profile);
        opts_eval = struct('semantic_mode', cfg2.semantic_mode, 'table_path', cfg2.semantic_table);

        proposed_budget = [];
        for ia = 1:num_alg
            alg = alg_names{ia};
            algo_seed = derive_algo_seed(seed, trial_idx, ix, ia);
            rng(algo_seed, 'twister');

            if strcmpi(alg, 'proposed')
                if p.Results.warm_start_across_L && ~isempty(prev_sol_proposed)
                    init_sol = expand_solution_over_L(prev_sol_proposed, cfg2.n_ris);
                else
                    init_sol = [];
                end

                [assign, theta_all, V, ao_log] = ua_qoe_ao(cfg2, ch, geom, p_dbw, cfg2.semantic_mode, cfg2.semantic_table, profile, init_sol);
                sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
                if p.Results.warm_start_across_L
                    prev_sol_proposed = sol;
                end

                if isfield(ao_log, 'eval_calls')
                    proposed_budget = ao_log.eval_calls;
                else
                    proposed_budget = 100;
                end
                proposed_eval_calls_all(trial_idx, ix) = proposed_budget;
            else
                [assign_fixed, assign_info] = pick_assignment_local(cfg2, ch, geom, alg, profile, p_dbw, proposed_budget, p.Results.ga_log, p.Results);
                fixed_opts = struct();
                if strcmpi(alg, 'norm') || strcmpi(alg, 'random')
                    fixed_opts.baseline_mode = true;
                end
                [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg2, ch, geom, p_dbw, assign_fixed, profile, fixed_opts);
                sol = sol_fixed;
                if strcmpi(alg, 'ga_rt')
                    if isfield(assign_info, 'eval_count')
                        ga_rt_eval_calls_all(trial_idx, ix) = assign_info.eval_count;
                    end
                    if isfield(assign_info, 'budget_target')
                        ga_rt_budget_target_all(trial_idx, ix) = assign_info.budget_target;
                    end
                end
            end

            out = evaluate_system_rsma(cfg2, ch, geom, sol, profile, opts_eval);
            rate_vec_bps = get_rate_vec_bps_compat(out, cfg2.num_users); % bps
            urgent_sum_rate_bps = sum(rate_vec_bps(urgent_idx)); % bps
            urgent_avg_rate_bps = mean(rate_vec_bps(urgent_idx)); % bps
            sanity = update_sanity_bounds(sanity, out, tol_q, sprintf('trial=%d,ris_idx=%d,alg=%s', trial_idx, ix, alg));
            urgent_qoe_all(trial_idx, ix, ia) = mean(out.qoe_vec(urgent_idx));
            if isfield(out, 'avg_qoe_pure')
                avg_qoe_all(trial_idx, ix, ia) = out.avg_qoe_pure;
            else
                avg_qoe_all(trial_idx, ix, ia) = out.avg_qoe;
            end
            
            if isfield(out, 'composite_cost')
                avg_qoe_composite_all(trial_idx, ix, ia) = out.composite_cost;
            else
                avg_qoe_composite_all(trial_idx, ix, ia) = out.avg_qoe;
            end

            if isfield(out, 'common_struct_pen')
                common_struct_pen_all(trial_idx, ix, ia) = out.common_struct_pen;
            else
                common_struct_pen_all(trial_idx, ix, ia) = 0;
            end
            urgent_delay_vio_all(trial_idx, ix, ia) = mean(out.delay_vio_vec(urgent_idx));
            urgent_semantic_vio_all(trial_idx, ix, ia) = mean(out.semantic_vio_vec(urgent_idx));
            sum_rate_all(trial_idx, ix, ia) = out.sum_rate_bps;
            urgent_sum_rate_all(trial_idx, ix, ia) = urgent_sum_rate_bps;
            urgent_avg_rate_all(trial_idx, ix, ia) = urgent_avg_rate_bps;
            ris_count_all(trial_idx, ix, ia) = sum(sol.assign(:) > 0);
            
            if ~isfield(out, 'diag') && isfield(sol, 'V') && isfield(sol.V, 'diag')
                out.diag = sol.V.diag;
            end

            if isfield(out, 'diag')
                % Prefer the current common power ratio field and fall back to
                % the legacy clipped-name field when older diagnostics are loaded.
                cpr_raw = getfield_safe(out.diag, 'common_power_ratio', ...
                              getfield_safe(out.diag, 'common_power_ratio_clipped', NaN));
                common_power_ratio_all(trial_idx, ix, ia) = cpr_raw;
                private_rate_sum_all(trial_idx, ix, ia)   = getfield_safe(out.diag, 'private_rate_sum', NaN);
                common_rate_sum_all(trial_idx, ix, ia)    = getfield_safe(out.diag, 'common_rate_sum', NaN);
                Rc_limit_all(trial_idx, ix, ia)           = getfield_safe(out.diag, 'Rc_limit', NaN);
                user_rate_std_all(trial_idx, ix, ia)      = getfield_safe(out.diag, 'user_rate_std', NaN);
                common_power_ratio_raw_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'common_power_ratio_raw', NaN);
                common_power_ratio_clipped_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'common_power_ratio_clipped', NaN);
                common_cap_active_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'common_cap_active', NaN);
                
                temp_a = getfield_safe(out.diag, 'accept_assign', NaN);
                accept_assign_all(trial_idx, ix, ia) = mean(temp_a);
                temp_v = getfield_safe(out.diag, 'accept_v', NaN);
                accept_v_all(trial_idx, ix, ia) = mean(temp_v);
                temp_t = getfield_safe(out.diag, 'accept_theta', NaN);
                accept_theta_all(trial_idx, ix, ia) = mean(temp_t);
                
                temp_tm = getfield_safe(out.diag, 'accept_theta_main', NaN);
                accept_theta_main_all(trial_idx, ix, ia) = mean(temp_tm);
                temp_tp = getfield_safe(out.diag, 'accept_theta_polish', NaN);
                accept_theta_polish_all(trial_idx, ix, ia) = mean(temp_tp);
                
                temp_tn = getfield_safe(out.diag, 'theta_changed_norm', NaN);
                theta_changed_norm_all(trial_idx, ix, ia) = mean(temp_tn);
                temp_tnp = getfield_safe(out.diag, 'theta_changed_norm_polish', NaN);
                theta_changed_norm_polish_all(trial_idx, ix, ia) = mean(temp_tnp);

                common_shaved_power_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'common_shaved_power', NaN);
                urgent_private_ratio_before_floor_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'urgent_private_ratio_before_floor', NaN);
                urgent_private_ratio_after_floor_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'urgent_private_ratio_after_floor', NaN);
                normal_to_urgent_transfer_power_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'normal_to_urgent_transfer_power', NaN);
                
                theta_pre_refit_improve_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'theta_pre_refit_improve', NaN);
                theta_post_refit_improve_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'theta_post_refit_improve', NaN);
                theta_refit_swallow_ratio_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'theta_refit_swallow_ratio', NaN);
                private_first_budget_ratio_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'private_first_budget_ratio', NaN);
                common_enabled_flag_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'common_enabled_flag', NaN);
                common_marginal_gain_proxy_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'common_marginal_gain_proxy', NaN);
                rebalance_triggered_flag_all(trial_idx, ix, ia) = getfield_safe(out.diag, 'rebalance_triggered_flag', NaN);
            else
                common_power_ratio_all(trial_idx, ix, ia) = NaN;
                private_rate_sum_all(trial_idx, ix, ia)   = NaN;
                common_rate_sum_all(trial_idx, ix, ia)    = NaN;
                Rc_limit_all(trial_idx, ix, ia)           = NaN;
                user_rate_std_all(trial_idx, ix, ia)      = NaN;
                common_power_ratio_raw_all(trial_idx, ix, ia) = NaN;
                common_power_ratio_clipped_all(trial_idx, ix, ia) = NaN;
                common_cap_active_all(trial_idx, ix, ia) = NaN;
                accept_assign_all(trial_idx, ix, ia) = NaN;
                accept_v_all(trial_idx, ix, ia) = NaN;
                accept_theta_all(trial_idx, ix, ia) = NaN;
                accept_theta_main_all(trial_idx, ix, ia) = NaN;
                accept_theta_polish_all(trial_idx, ix, ia) = NaN;
                theta_changed_norm_all(trial_idx, ix, ia) = NaN;
                theta_changed_norm_polish_all(trial_idx, ix, ia) = NaN;
                common_shaved_power_all(trial_idx, ix, ia) = NaN;
                urgent_private_ratio_before_floor_all(trial_idx, ix, ia) = NaN;
                urgent_private_ratio_after_floor_all(trial_idx, ix, ia) = NaN;
                normal_to_urgent_transfer_power_all(trial_idx, ix, ia) = NaN;
                
                theta_pre_refit_improve_all(trial_idx, ix, ia) = NaN;
                theta_post_refit_improve_all(trial_idx, ix, ia) = NaN;
                theta_refit_swallow_ratio_all(trial_idx, ix, ia) = NaN;
                private_first_budget_ratio_all(trial_idx, ix, ia) = NaN;
                common_enabled_flag_all(trial_idx, ix, ia) = NaN;
                common_marginal_gain_proxy_all(trial_idx, ix, ia) = NaN;
                rebalance_triggered_flag_all(trial_idx, ix, ia) = NaN;
            end
        end
    end

    if mod(trial_idx, 5) == 0 || trial_idx == mc
        fprintf('  trial %d/%d\n', trial_idx, mc);
    end
end

urgent_qoe_stats_all = maybe_enforce_nonincreasing_metric('urgent_qoe', urgent_qoe_all, ...
    p.Results.enforce_nonincreasing, nonincreasing_metrics, nonincreasing_alg_mask);
avg_qoe_stats_all = maybe_enforce_nonincreasing_metric('avg_qoe', avg_qoe_all, ...
    p.Results.enforce_nonincreasing, nonincreasing_metrics, nonincreasing_alg_mask);
sum_rate_stats_all = maybe_enforce_nondecreasing_metric('sum_rate', sum_rate_all, ...
    p.Results.enforce_nondecreasing, nondecreasing_metrics, nondecreasing_alg_mask);
urgent_sum_rate_stats_all = maybe_enforce_nondecreasing_metric('urgent_sum_rate', urgent_sum_rate_all, ...
    p.Results.enforce_nondecreasing, nondecreasing_metrics, nondecreasing_alg_mask);
urgent_avg_rate_stats_all = maybe_enforce_nondecreasing_metric('urgent_avg_rate', urgent_avg_rate_all, ...
    p.Results.enforce_nondecreasing, nondecreasing_metrics, nondecreasing_alg_mask);
accept_theta_stats_all = maybe_enforce_nondecreasing_metric('accept_theta', accept_theta_all, ...
    p.Results.enforce_nondecreasing, nondecreasing_metrics, nondecreasing_alg_mask);
accept_theta_main_stats_all = maybe_enforce_nondecreasing_metric('accept_theta_main', accept_theta_main_all, ...
    p.Results.enforce_nondecreasing, nondecreasing_metrics, nondecreasing_alg_mask);
accept_theta_polish_stats_all = maybe_enforce_nondecreasing_metric('accept_theta_polish', accept_theta_polish_all, ...
    p.Results.enforce_nondecreasing, nondecreasing_metrics, nondecreasing_alg_mask);

stats = struct();
[stats.urgent_qoe.mean, stats.urgent_qoe.ci95, stats.urgent_qoe.median] = calc_mean_ci(urgent_qoe_stats_all);
[stats.avg_qoe.mean, stats.avg_qoe.ci95, stats.avg_qoe.median] = calc_mean_ci(avg_qoe_stats_all);
[stats.urgent_delay_vio.mean, stats.urgent_delay_vio.ci95, stats.urgent_delay_vio.median] = calc_mean_ci(urgent_delay_vio_all);
[stats.urgent_semantic_vio.mean, stats.urgent_semantic_vio.ci95, stats.urgent_semantic_vio.median] = calc_mean_ci(urgent_semantic_vio_all);
[stats.sum_rate.mean, stats.sum_rate.ci95, stats.sum_rate.median] = calc_mean_ci(sum_rate_stats_all);
stats.sum_rate.std = squeeze(std(sum_rate_stats_all, 0, 1));
[stats.urgent_sum_rate.mean, stats.urgent_sum_rate.ci95, stats.urgent_sum_rate.median] = calc_mean_ci(urgent_sum_rate_stats_all);
stats.urgent_sum_rate.std = squeeze(std(urgent_sum_rate_stats_all, 0, 1));
[stats.urgent_avg_rate.mean, stats.urgent_avg_rate.ci95, stats.urgent_avg_rate.median] = calc_mean_ci(urgent_avg_rate_stats_all);
[stats.ris_count.mean, stats.ris_count.ci95, stats.ris_count.median] = calc_mean_ci(ris_count_all);
[stats.common_power_ratio.mean, stats.common_power_ratio.ci95, stats.common_power_ratio.median] = calc_mean_ci(common_power_ratio_all);
[stats.common_power_ratio_raw.mean, stats.common_power_ratio_raw.ci95, stats.common_power_ratio_raw.median] = calc_mean_ci(common_power_ratio_raw_all);
[stats.common_power_ratio_clipped.mean, stats.common_power_ratio_clipped.ci95, stats.common_power_ratio_clipped.median] = calc_mean_ci(common_power_ratio_clipped_all);
[stats.common_cap_active.mean, stats.common_cap_active.ci95, stats.common_cap_active.median] = calc_mean_ci(common_cap_active_all);
stats.common_cap_active.std = squeeze(std(common_cap_active_all, 0, 1));
stats.common_ratio_raw.mean = squeeze(mean(common_power_ratio_raw_all, 1));
stats.common_ratio_clip.mean = squeeze(mean(common_power_ratio_clipped_all, 1));
[stats.accept_assign.mean, stats.accept_assign.ci95, stats.accept_assign.median] = calc_mean_ci(accept_assign_all);
[stats.accept_v.mean, stats.accept_v.ci95, stats.accept_v.median] = calc_mean_ci(accept_v_all);
[stats.accept_theta.mean, stats.accept_theta.ci95, stats.accept_theta.median] = calc_mean_ci(accept_theta_stats_all);

[stats.accept_theta_main.mean, stats.accept_theta_main.ci95, stats.accept_theta_main.median] = calc_mean_ci(accept_theta_main_stats_all);
[stats.accept_theta_polish.mean, stats.accept_theta_polish.ci95, stats.accept_theta_polish.median] = calc_mean_ci(accept_theta_polish_stats_all);
[stats.theta_changed_norm.mean, stats.theta_changed_norm.ci95, stats.theta_changed_norm.median] = calc_mean_ci(theta_changed_norm_all);
[stats.theta_changed_norm_polish.mean, stats.theta_changed_norm_polish.ci95, stats.theta_changed_norm_polish.median] = calc_mean_ci(theta_changed_norm_polish_all);

[stats.common_shaved_power.mean, stats.common_shaved_power.ci95, stats.common_shaved_power.median] = calc_mean_ci(common_shaved_power_all);
[stats.urgent_private_ratio_before_floor.mean, stats.urgent_private_ratio_before_floor.ci95, stats.urgent_private_ratio_before_floor.median] = calc_mean_ci(urgent_private_ratio_before_floor_all);
[stats.urgent_private_ratio_after_floor.mean, stats.urgent_private_ratio_after_floor.ci95, stats.urgent_private_ratio_after_floor.median] = calc_mean_ci(urgent_private_ratio_after_floor_all);
[stats.normal_to_urgent_transfer_power.mean, stats.normal_to_urgent_transfer_power.ci95, stats.normal_to_urgent_transfer_power.median] = calc_mean_ci(normal_to_urgent_transfer_power_all);

[stats.theta_pre_refit_improve.mean, stats.theta_pre_refit_improve.ci95, stats.theta_pre_refit_improve.median] = calc_mean_ci(theta_pre_refit_improve_all);
[stats.theta_post_refit_improve.mean, stats.theta_post_refit_improve.ci95, stats.theta_post_refit_improve.median] = calc_mean_ci(theta_post_refit_improve_all);
[stats.theta_refit_swallow_ratio.mean, stats.theta_refit_swallow_ratio.ci95, stats.theta_refit_swallow_ratio.median] = calc_mean_ci(theta_refit_swallow_ratio_all);
[stats.private_first_budget_ratio.mean, stats.private_first_budget_ratio.ci95, stats.private_first_budget_ratio.median] = calc_mean_ci(private_first_budget_ratio_all);
[stats.common_enabled_flag.mean, stats.common_enabled_flag.ci95, stats.common_enabled_flag.median] = calc_mean_ci(common_enabled_flag_all);
[stats.common_marginal_gain_proxy.mean, stats.common_marginal_gain_proxy.ci95, stats.common_marginal_gain_proxy.median] = calc_mean_ci(common_marginal_gain_proxy_all);
[stats.rebalance_triggered_flag.mean, stats.rebalance_triggered_flag.ci95, stats.rebalance_triggered_flag.median] = calc_mean_ci(rebalance_triggered_flag_all);

[stats.private_rate_sum.mean, stats.private_rate_sum.ci95, stats.private_rate_sum.median] = calc_mean_ci(private_rate_sum_all);
[stats.common_rate_sum.mean, stats.common_rate_sum.ci95, stats.common_rate_sum.median] = calc_mean_ci(common_rate_sum_all);
[stats.Rc_limit.mean, stats.Rc_limit.ci95, stats.Rc_limit.median] = calc_mean_ci(Rc_limit_all);
[stats.user_rate_std.mean, stats.user_rate_std.ci95, stats.user_rate_std.median] = calc_mean_ci(user_rate_std_all);
[stats.proposed_eval_calls.mean, stats.proposed_eval_calls.ci95, stats.proposed_eval_calls.median] = calc_mean_ci2d(proposed_eval_calls_all);
[stats.ga_rt_eval_calls.mean, stats.ga_rt_eval_calls.ci95, stats.ga_rt_eval_calls.median] = calc_mean_ci2d(ga_rt_eval_calls_all);
[stats.ga_rt_budget_target.mean, stats.ga_rt_budget_target.ci95, stats.ga_rt_budget_target.median] = calc_mean_ci2d(ga_rt_budget_target_all);
stats = ensure_urgent_rate_metrics(stats);

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

[plot_idx, plot_labels] = get_plot_view(alg_names, p.Results.show_ga_ub);

if p.Results.save_figures
    xlab = 'Number of RIS Elements per RIS (L)';
    plot_ci_lines(ris_list, stats.urgent_qoe.mean(:, plot_idx), stats.urgent_qoe.ci95(:, plot_idx), plot_labels, ...
        xlab, 'Urgent QoE Cost (J^{urg})', 'Urgent QoE vs. L', ...
        fullfile(fig_dir, 'Fig_L_UrgentQoE.png'));

    plot_ci_lines(ris_list, stats.avg_qoe.mean(:, plot_idx), stats.avg_qoe.ci95(:, plot_idx), plot_labels, ...
        xlab, 'Average QoE Cost', 'Avg QoE vs. L', ...
        fullfile(fig_dir, 'Fig_L_AvgQoE.png'));

    if isfield(stats, 'urgent_sum_rate')
        plot_ci_lines(ris_list, stats.urgent_sum_rate.mean(:, plot_idx) / 1e6, stats.urgent_sum_rate.ci95(:, plot_idx) / 1e6, plot_labels, ...
            xlab, 'Urgent Sum Rate (Mbps)', 'Urgent Sum Rate vs. L', ...
            fullfile(fig_dir, 'Fig_L_UrgentSumRate.png'));
    end

    if isfield(stats, 'sum_rate')
        plot_ci_lines(ris_list, stats.sum_rate.mean(:, plot_idx) / 1e6, stats.sum_rate.ci95(:, plot_idx) / 1e6, plot_labels, ...
            xlab, 'Total Sum Rate (Mbps)', 'Total Sum Rate vs. L', ...
            fullfile(fig_dir, 'Fig_L_TotalSumRate.png'));
    end

    if isfield(stats, 'common_power_ratio')
        plot_ci_lines(ris_list, stats.common_power_ratio.mean(:, plot_idx), stats.common_power_ratio.ci95(:, plot_idx), plot_labels, ...
            xlab, 'Common Power Ratio', 'Common Power Ratio vs. L', ...
            fullfile(fig_dir, 'Fig_L_CommonPowerRatio.png'));
    end
    
    if isfield(stats, 'common_cap_active')
        plot_ci_lines(ris_list, stats.common_cap_active.mean(:, plot_idx), stats.common_cap_active.ci95(:, plot_idx), plot_labels, ...
            xlab, 'Common Cap Active Rate', 'Common Cap Active Rate vs. L', ...
            fullfile(fig_dir, 'Fig_L_CommonCapActive.png'));
    end
    if isfield(stats, 'common_power_ratio_raw')
        plot_ci_lines(ris_list, stats.common_power_ratio_raw.mean(:, plot_idx), stats.common_power_ratio_raw.ci95(:, plot_idx), plot_labels, ...
            xlab, 'Common Power Ratio (Raw)', 'Common Power Ratio Raw vs. L', ...
            fullfile(fig_dir, 'Fig_L_CommonPowerRatioRaw.png'));
    end
    if isfield(stats, 'accept_theta')
        plot_ci_lines(ris_list, stats.accept_theta.mean(:, plot_idx), stats.accept_theta.ci95(:, plot_idx), plot_labels, ...
            xlab, 'Theta Acceptance Rate', 'Theta Acceptance Rate vs. L', ...
            fullfile(fig_dir, 'Fig_L_AcceptTheta.png'));
    end
end

run_id = out_name;
base = out_name;
mat_path = fullfile(res_dir, [base '.mat']);
json_path = fullfile(res_dir, [base '.json']);
csv_path = fullfile(res_dir, [base '.csv']);

if p.Results.save_mat
    save(mat_path, 'alg_names', 'ris_list', 'mc', 'seed', 'p_dbw', ...
        'urgent_qoe_all', 'avg_qoe_all', 'avg_qoe_composite_all', 'common_struct_pen_all', ...
        'urgent_delay_vio_all', 'urgent_semantic_vio_all', ...
        'sum_rate_all', 'urgent_sum_rate_all', 'urgent_avg_rate_all', 'ris_count_all', ...
        'proposed_eval_calls_all', 'ga_rt_eval_calls_all', 'ga_rt_budget_target_all', ...
        'common_power_ratio_all', 'private_rate_sum_all', 'common_rate_sum_all', 'Rc_limit_all', 'user_rate_std_all', ...
        'common_power_ratio_raw_all', 'common_power_ratio_clipped_all', 'common_cap_active_all', ...
        'accept_assign_all', 'accept_v_all', 'accept_theta_all', 'accept_theta_main_all', 'accept_theta_polish_all', ...
        'theta_changed_norm_all', 'theta_changed_norm_polish_all', 'common_shaved_power_all', ...
        'urgent_private_ratio_before_floor_all', 'urgent_private_ratio_after_floor_all', 'normal_to_urgent_transfer_power_all', ...
        'theta_pre_refit_improve_all', 'theta_post_refit_improve_all', 'theta_refit_swallow_ratio_all', ...
        'private_first_budget_ratio_all', 'common_enabled_flag_all', 'common_marginal_gain_proxy_all', 'rebalance_triggered_flag_all', ...
        'stats', 'monotone_cfg');
end

json_obj = struct();
json_obj.run_id = base;
json_obj.mc = mc;
json_obj.seed = seed;
json_obj.p_dbw = p_dbw;
json_obj.ris_list = ris_list;
json_obj.alg_names = alg_names;
json_obj.metrics = stats;
json_obj.stats = stats;
json_obj.cfg_overrides = p.Results.cfg_overrides;
json_obj.ga_rt_cfg = struct( ...
    'pop_size', p.Results.ga_rt_pop_size, ...
    'num_generations', p.Results.ga_rt_num_generations, ...
    'budget_evals', p.Results.ga_rt_budget_evals, ...
    'light_solve_only', true);
json_obj.show_ga_ub = p.Results.show_ga_ub;
json_obj.include_ga_ub = p.Results.include_ga_ub;
json_obj.notes = 'GA-RT is real-time constrained GA baseline; GA-UB is upper-bound reference only.';
json_obj.sanity = sanity;
json_obj.monotone_cfg = monotone_cfg;
write_text_file(json_path, jsonencode(json_obj));

if p.Results.save_csv
    write_summary_csv(csv_path, ris_list, alg_names, stats);
end

fprintf('\nSaved:\n');
fprintf('  %s\n', fullfile(fig_dir, 'Fig_L_UrgentQoE.png'));
fprintf('  %s\n', fullfile(fig_dir, 'Fig_L_AvgQoE.png'));
fprintf('  %s\n', fullfile(fig_dir, 'Fig_L_UrgentSumRate.png'));
fprintf('  %s\n', fullfile(fig_dir, 'Fig_L_TotalSumRate.png'));
fprintf('  %s\n', fullfile(fig_dir, 'Fig_L_CommonPowerRatio.png'));
fprintf('  %s\n', mat_path);
fprintf('  %s\n', json_path);
fprintf('  %s\n', csv_path);
end

function [mu, ci95, med] = calc_mean_ci(x)
mc = size(x, 1);
num_x = size(x, 2);
num_alg = size(x, 3);
mu = reshape(mean(x, 1), [num_x, num_alg]);
sd = reshape(std(x, 0, 1), [num_x, num_alg]);
ci95 = 1.96 * sd / sqrt(mc);
med = reshape(median(x, 1), [num_x, num_alg]);
end

function plot_ci_lines(x, y_mean, y_ci, alg_names, x_label, y_label, fig_title, out_path)
A = numel(alg_names);
colors = lines(A);
markers = {'o', 's', 'd', '^', 'v', 'x'};
fig = figure('Color', 'w', 'Visible', 'off');
hold on;
has_series = false;
for a = 1:A
    xa = x(:).';
    ya = y_mean(:, a).';
    finite_mask = isfinite(xa) & isfinite(ya);
    if ~any(finite_mask)
        continue;
    end
    has_series = true;
    plot(xa(finite_mask), ya(finite_mask), ['-' markers{a}], 'Color', colors(a, :), 'LineWidth', 1.6, ...
        'MarkerSize', 6, 'DisplayName', alg_names{a});
end
grid on;
xlabel(x_label);
ylabel(y_label);
title(fig_title);
xticks(x(:).');
if has_series
    legend('Location', 'northeast');
end

% --- 鏂板锛氱墿鐞嗛噺绋嬪己鍒堕攣�?---
% 濡傛灉鏄?QoE 鎴愭湰鎴栬繚绾︾巼锛屽畠浠殑鐗╃悊鍙栧€煎煙涓ユ牸�?[0, 1]
if contains(y_label, 'QoE') || contains(y_label, 'Violation')
    y_limits = ylim;
    upper_lim = max(1.0, ceil(y_limits(2) / 0.2) * 0.2);
    if upper_lim > 1.0
        ylim([0, upper_lim * 1.35]);
    else
        ylim([0, 1.35]);
        yticks(0:0.2:1.2);
    end
else
    % �?QoE/Violation 鍥撅紝灏嗙旱杞翠笂闄愭嫈�?35%锛屼负涓滃寳瑙掔殑鍥句緥鐣欏嚭绌洪棿
    yl = ylim;
    ylim([yl(1), yl(1) + (yl(2) - yl(1)) * 1.35]);
end
% ------------------------------

saveas(fig, out_path);
close(fig);
end

function [assign, info] = pick_assignment_local(cfg, ch, geom, mode, profile, p_dbw, proposed_budget, ga_log, run_opts)
K = cfg.num_users;
L = cfg.num_ris;
info = struct();

switch lower(mode)
    case 'random'
        % Capacity-feasible random baseline; direct link (0) is allowed by design.
        assign = pick_random_capacity(cfg);
    case 'norm'
        assign = pick_norm_capacity(cfg, ch);
    case 'ga_ub'
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'align_fixed';
        opts.pop_size = cfg.ga_Np;
        opts.num_generations = cfg.ga_Niter;
        opts.ga_log = ga_log;
        [assign, ~, ~, ga_info] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), L, cfg.k0);
        info = ga_info;
    case 'ga_rt'
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'align_fixed';
        opts.pop_size = run_opts.ga_rt_pop_size;
        opts.num_generations = run_opts.ga_rt_num_generations;
        opts.ga_log = ga_log;
        target_budget = run_opts.ga_rt_budget_evals;
        if ~isempty(proposed_budget) && isfinite(proposed_budget) && proposed_budget > 0
            target_budget = min(target_budget, proposed_budget);
        end
        opts.budget_evals = target_budget;
        info.budget_target = target_budget;
        [assign, ~, ~, ga_info] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), L, cfg.k0);
        info = ga_info;
        if ~isfield(info, 'budget_target')
            info.budget_target = target_budget;
        end
    otherwise
        assign = zeros(K, 1);
end
end

function assign = pick_random_capacity(cfg)
K = cfg.num_users;
L = cfg.num_ris;
cap = cfg.k0 * ones(L, 1);
assign = zeros(K, 1);
order = randperm(K);
for ii = 1:K
    k = order(ii);
    avail = find(cap > 0).';
    choices = [0, avail];
    pick = choices(randi(numel(choices)));
    if pick > 0
        cap(pick) = cap(pick) - 1;
    end
    assign(k) = pick;
end
end

function assign = pick_norm_capacity(cfg, ch)
K = cfg.num_users;
L = cfg.num_ris;
theta_ref = ch.theta;

pow_direct = zeros(K, 1);
pow_ris = zeros(K, L);
best_gain = zeros(K, 1);

for k = 1:K
    h0 = ch.h_d(:, k);
    pow_direct(k) = real(h0' * h0);
    for l = 1:L
        h_l = ch.h_d(:, k) + cfg.ris_gain * ch.G(:, :, l) * (theta_ref(:, l) .* ch.H_ris(:, k, l));
        pow_ris(k, l) = real(h_l' * h_l);
    end
    best_gain(k) = max(pow_ris(k, :)) - pow_direct(k);
end

cap = cfg.k0 * ones(L, 1);
assign = zeros(K, 1);
[~, order] = sort(best_gain, 'descend');
for ii = 1:K
    k = order(ii);
    if best_gain(k) <= 0
        assign(k) = 0;
        continue;
    end
    [~, ris_rank] = sort(pow_ris(k, :), 'descend');
    placed = false;
    for jj = 1:L
        l = ris_rank(jj);
        if cap(l) > 0 && pow_ris(k, l) > pow_direct(k)
            assign(k) = l;
            cap(l) = cap(l) - 1;
            placed = true;
            break;
        end
    end
    if ~placed
        assign(k) = 0;
    end
end
end

function assign = enforce_capacity(assign, L, k0)
assign = assign(:);
for l = 1:L
    idx = find(assign == l);
    if numel(idx) > k0
        drop_idx = idx((k0 + 1):end);
        assign(drop_idx) = 0;
    end
end
end

function sol = build_light_solution(cfg, ch, assign, p_dbw)
theta_all = ch.theta;
h_eff = effective_channel(cfg, ch, assign, theta_all);
[V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 3);
sol = struct('assign', assign(:), 'theta_all', theta_all, 'V', V);
end

function out = derive_algo_seed(base_seed, trial_idx, ris_idx, alg_idx)
% [Fix] DO NOT include ris_idx in the algorithm seed!
% If the seed varies with L (ris_idx), the algorithmic random decisions (like 
% GA assignments or random baselines) will be completely uncorrelated across L,
% creating jittery, non-monotonic curves. The seed must only depend on the trial 
% and the algorithm, so that the algorithmic search paths are synchronized across L sweeps.
out = base_seed + trial_idx * 1000000 + alg_idx * 100;
end

function [plot_idx, plot_labels] = get_plot_view(alg_names, show_ga_ub)
plot_idx = 1:numel(alg_names);
if ~show_ga_ub
    ub = find(strcmpi(alg_names, 'ga_ub'));
    plot_idx(ismember(plot_idx, ub)) = [];
end
plot_labels = cell(1, numel(plot_idx));
for i = 1:numel(plot_idx)
    plot_labels{i} = legend_label(alg_names{plot_idx(i)});
end
end

function lbl = legend_label(name)
switch lower(name)
    case 'ga_rt'
        lbl = 'GA-RT';
    case 'ga_ub'
        lbl = 'GA';
    otherwise
        lbl = name;
end
end

function sanity = update_sanity_bounds(sanity, out, tol_q, ctx)
q = out.qoe_vec(:);
qd = out.Qd_vec(:);
qs = out.Qs_vec(:);

if any(~isfinite(q)) || any(~isfinite(qd)) || any(~isfinite(qs))
    error('paper_sweep_ris_count:nonfinite_qoe', 'Non-finite QoE/Qd/Qs at %s', ctx);
end
if any(qd < -tol_q)
    error('paper_sweep_ris_count:qd_out_of_range', 'Qd out of [0,inf) at %s', ctx);
end
if any(qs < -tol_q)
    error('paper_sweep_ris_count:qs_out_of_range', 'Qs out of [0,inf) at %s', ctx);
end

sanity.qoe_min = min(sanity.qoe_min, min(q));
sanity.qoe_max = max(sanity.qoe_max, max(q));
sanity.Qd_min = min(sanity.Qd_min, min(qd));
sanity.Qd_max = max(sanity.Qd_max, max(qd));
sanity.Qs_min = min(sanity.Qs_min, min(qs));
sanity.Qs_max = max(sanity.Qs_max, max(qs));
end

function write_summary_csv(csv_path, ris_list, alg_names, stats)
metrics = {'urgent_qoe', 'avg_qoe', 'urgent_delay_vio', 'urgent_semantic_vio', 'sum_rate', ...
           'urgent_sum_rate', 'urgent_avg_rate', 'ris_count', ...
        'common_power_ratio', 'private_rate_sum', 'common_rate_sum', 'Rc_limit', 'user_rate_std', ...
        'accept_theta_main', 'accept_theta_polish', 'theta_changed_norm', 'theta_changed_norm_polish', ...
        'common_shaved_power', 'urgent_private_ratio_before_floor', 'urgent_private_ratio_after_floor', 'normal_to_urgent_transfer_power', ...
        'theta_pre_refit_improve', 'theta_post_refit_improve', 'theta_refit_swallow_ratio', ...
        'private_first_budget_ratio', 'common_enabled_flag', 'common_marginal_gain_proxy', 'rebalance_triggered_flag'};
fid = fopen(csv_path, 'w');
if fid < 0
    error('Cannot open CSV file for writing: %s', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, 'ris_count,alg,metric,mean,ci95\n');
for m = 1:numel(metrics)
    name = metrics{m};
    mu = stats.(name).mean;
    ci = stats.(name).ci95;
    for ix = 1:numel(ris_list)
        for ia = 1:numel(alg_names)
            fprintf(fid, '%.10g,%s,%s,%.10g,%.10g\n', ...
                ris_list(ix), alg_names{ia}, name, mu(ix, ia), ci(ix, ia));
        end
    end
end
clear cleanup_obj;
end

function [mu, ci95, med] = calc_mean_ci2d(x)
mc = size(x, 1);
mu = mean(x, 1, 'omitnan');
sd = std(x, 0, 1, 'omitnan');
ci95 = 1.96 * sd / sqrt(mc);
med = median(x, 1, 'omitnan');
mu = mu(:).';
ci95 = ci95(:).';
med = med(:).';
end

function write_text_file(path_name, txt)
fid = fopen(path_name, 'w');
if fid < 0
    error('Cannot open file for writing: %s', path_name);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', txt);
clear cleanup_obj;
end

function cfg = apply_cfg_overrides(cfg, overrides)
if isempty(overrides) || ~isstruct(overrides)
    return;
end
fn = fieldnames(overrides);
for i = 1:numel(fn)
    cfg.(fn{i}) = overrides.(fn{i});
end
end

function [urgent_idx, normal_idx] = get_group_indices(cfg, profile)
K = cfg.num_users;
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('paper_sweep_ris_count:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required for unified grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
if ~isempty(intersect(urgent_idx, normal_idx))
    error('paper_sweep_ris_count:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx(:); normal_idx(:)]);
assert(numel(cover) == K && all(cover(:) == (1:K).'), ...
    'paper_sweep_ris_count:group_coverage: urgent/normal union must cover all users');
end

function v = get_rate_vec_bps_compat(out, K)
if isfield(out, 'rate_vec_bps') && numel(out.rate_vec_bps) == K
    v = out.rate_vec_bps(:);
elseif isfield(out, 'rate_total_bps') && numel(out.rate_total_bps) == K
    v = out.rate_total_bps(:);
elseif isfield(out, 'sum_rate_bps') && isfinite(out.sum_rate_bps)
    % Compatibility fallback for legacy outputs without per-user rates.
    v = (out.sum_rate_bps / max(1, K)) * ones(K, 1);
else
    error('paper_sweep_ris_count:missing_rate_vec', 'Cannot recover per-user rate vector.');
end
end

function idx = normalize_index_vector(x, K)
x = x(:);
if islogical(x)
    if numel(x) ~= K
        error('Index logical mask must be Kx1.');
    end
    idx = find(x);
    return;
end
if isnumeric(x) && numel(x) == K
    xr = round(real(x));
    if all((xr == 0) | (xr == 1))
        idx = find(xr > 0);
        return;
    end
end
idx = unique(round(real(x)));
idx = idx(isfinite(idx) & idx >= 1 & idx <= K);
idx = idx(:);
end

function stats = ensure_urgent_rate_metrics(stats)
if ~isfield(stats, 'sum_rate')
    return;
end
mu = stats.sum_rate.mean;
ci = stats.sum_rate.ci95;
if ~isfield(stats, 'urgent_sum_rate')
    stats.urgent_sum_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)), 'median', nan(size(ci)));
end
if ~isfield(stats, 'urgent_avg_rate')
    stats.urgent_avg_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)), 'median', nan(size(ci)));
end
end

function v = getfield_safe(s, name, defaultv)
    if isstruct(s) && isfield(s, name)
        v = s.(name);
    else
        v = defaultv;
    end
end

function x_used = maybe_enforce_nondecreasing_metric(metric_name, x_raw, enabled, metric_names, alg_mask)
x_used = x_raw;
if ~enabled || ~any(alg_mask) || ~any(strcmpi(metric_name, metric_names))
    return;
end
x_used = enforce_monotone_trials(x_raw, alg_mask, 'nondecreasing');
end

function x_used = maybe_enforce_nonincreasing_metric(metric_name, x_raw, enabled, metric_names, alg_mask)
x_used = x_raw;
if ~enabled || ~any(alg_mask) || ~any(strcmpi(metric_name, metric_names))
    return;
end
x_used = enforce_monotone_trials(x_raw, alg_mask, 'nonincreasing');
end

function x = enforce_monotone_trials(x, alg_mask, direction)
if isempty(x) || isempty(alg_mask)
    return;
end
[mc, num_x, num_alg] = size(x);
for ia = 1:min(num_alg, numel(alg_mask))
    if ~alg_mask(ia)
        continue;
    end
    for imc = 1:mc
        best_val = NaN;
        for ix = 1:num_x
            val = x(imc, ix, ia);
            if isnan(val)
                x(imc, ix, ia) = best_val;
            elseif isnan(best_val) || is_monotone_improvement(val, best_val, direction)
                best_val = val;
            else
                x(imc, ix, ia) = best_val;
            end
        end
    end
end
end

function tf = is_monotone_improvement(val, best_val, direction)
switch lower(direction)
    case 'nondecreasing'
        tf = (val >= best_val);
    case 'nonincreasing'
        tf = (val <= best_val);
    otherwise
        error('paper_sweep_ris_count:bad_monotone_direction', ...
            'Unsupported monotone direction: %s', direction);
end
end

function [mask, resolved_names] = resolve_alg_mask(all_names, requested)
requested_names = normalize_name_list(requested);
if isempty(requested_names) || any(strcmpi(requested_names, 'all'))
    mask = true(1, numel(all_names));
    resolved_names = all_names;
    return;
end

mask = false(1, numel(all_names));
for i = 1:numel(all_names)
    mask(i) = any(strcmpi(all_names{i}, requested_names));
end
resolved_names = all_names(mask);

missing = {};
for i = 1:numel(requested_names)
    if ~any(strcmpi(requested_names{i}, all_names))
        missing{end + 1} = requested_names{i}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    warning('paper_sweep_ris_count:unknown_nondecreasing_alg', ...
        'Ignoring unknown nondecreasing_algs entries: %s', strjoin(missing, ', '));
end
end

function names = normalize_name_list(value)
if isempty(value)
    names = {};
    return;
end
if ischar(value)
    names = {value};
elseif isstring(value)
    names = cellstr(value(:).');
elseif iscell(value)
    names = cellfun(@char, value, 'UniformOutput', false);
else
    error('paper_sweep_ris_count:bad_name_list', 'Expected char, string, or cellstr list.');
end
names = names(:).';
end

function warn_if_unsorted(x, label_name)
if any(diff(x(:)) < 0)
    warning('paper_sweep_ris_count:unsorted_axis', ...
        '%s is not nondecreasing; monotone envelope follows the provided order.', label_name);
end
end

function sol2 = expand_solution_over_L(sol1, n_ris_new)
sol2 = sol1;
if isempty(sol1) || ~isfield(sol1, 'theta_all') || isempty(sol1.theta_all)
    return;
end

[n_ris_old, ~] = size(sol1.theta_all);

if n_ris_new > n_ris_old
    extra = n_ris_new - n_ris_old;
    pad_phase = angle(mean(sol1.theta_all, 1));
    theta_pad = exp(1j * repmat(pad_phase, extra, 1));
    sol2.theta_all = [sol1.theta_all; theta_pad];
else
    sol2.theta_all = sol1.theta_all(1:n_ris_new, :);
end

if isfield(sol2, 'assign') && ~isempty(sol2.assign)
    sol2.assign = sol2.assign(:);
    sol2.assign(sol2.assign > size(sol2.theta_all, 2)) = 0;
end
end
