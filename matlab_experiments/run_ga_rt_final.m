function run_ga_rt_final()
%RUN_GA_RT_FINAL Final paper-ready closed loop under GA-RT baseline.
% Outputs:
%   results/paper_sweep_power_ga_rt_final.json
%   results/ris_count_ga_rt_final.json
%   results/final_report.md

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg0 = config();

seeds_full = [40 41 42 43 44];
mc_per_seed = 30;
p_list = [-25, -22.14, -19.29, -16.43, -13.57, -10.71, -7.86, -5];
ris_list = [1 2 3 4];
p_dbw_ris = -8;
alg_names = {'proposed', 'random', 'norm', 'ga_rt'};

cfg_rt = struct('pop_size', 8, 'num_generations', 6, 'budget_evals', 40, 'light_solve_only', true);

% Warm-start hard constraint.
warm_start = false;

% --------- Lightweight tuning (cold-start only) ---------
seeds_tune = [40 41 42];
mc_tune = 6;
seeds_val = [40 41 42 43 44];
mc_val = 12;
cands = build_candidates_final();
best = [];
best_score = -inf;
mini_scores = -inf(1, numel(cands));
fprintf('=== Final tuning (cold-start only): %d candidates ===\n', numel(cands));

for i = 1:numel(cands)
    cand = cands(i);
    tmp = run_core(cfg0, cand.cfg, seeds_tune, mc_tune, p_list, ris_list, p_dbw_ris, alg_names, cfg_rt, false);
    ev = evaluate_candidate(tmp.power, tmp.ris, alg_names, p_list, ris_list);
    fprintf('  cand %02d score=%.6f pass=%d mean(dQ)=%.6f mean(dS)=%.6f max(dD)=%.6f\n', ...
        i, ev.score, ev.pass, ev.mean_delta_qoe, ev.mean_delta_sem, ev.max_delta_delay);
    mini_scores(i) = ev.score;
end

[~, ord] = sort(mini_scores, 'descend');
topk = ord(1:min(5, numel(ord)));
fprintf('=== Validation on larger subset: top-%d, seeds=%s, mc_each=%d ===\n', numel(topk), mat2str(seeds_val), mc_val);
for ii = 1:numel(topk)
    i = topk(ii);
    cand = cands(i);
    tmp = run_core(cfg0, cand.cfg, seeds_val, mc_val, p_list, ris_list, p_dbw_ris, alg_names, cfg_rt, false);
    ev = evaluate_candidate(tmp.power, tmp.ris, alg_names, p_list, ris_list);
    fprintf('  val cand %02d score=%.6f pass=%d mean(dQ)=%.6f mean(dS)=%.6f max(dD)=%.6f\n', ...
        i, ev.score, ev.pass, ev.mean_delta_qoe, ev.mean_delta_sem, ev.max_delta_delay);
    if ev.pass && ev.score > best_score
        best_score = ev.score;
        best = struct('id', i, 'cfg', cand.cfg, 'eval', ev);
    end
end

if isempty(best)
    fprintf('No candidate passed strict delay guard, fallback to minimum max delay violation.\n');
    min_bad = inf;
    best_local = 1;
    best_ev = [];
    for i = 1:numel(cands)
        cand = cands(i);
        tmp = run_core(cfg0, cand.cfg, seeds_tune, mc_tune, p_list, ris_list, p_dbw_ris, alg_names, cfg_rt, false);
        ev = evaluate_candidate(tmp.power, tmp.ris, alg_names, p_list, ris_list);
        bad = max(0, ev.max_delta_delay - 0.02);
        if bad < min_bad
            min_bad = bad;
            best_local = i;
            best_ev = ev;
        end
    end
    best = struct('id', best_local, 'cfg', cands(best_local).cfg, 'eval', best_ev);
end

fprintf('Chosen candidate: %d\n', best.id);

% --------- Full run ---------
out = run_core(cfg0, best.cfg, seeds_full, mc_per_seed, p_list, ris_list, p_dbw_ris, alg_names, cfg_rt, true);
N_total = numel(seeds_full) * mc_per_seed;

stats_power = build_stats_power(out.power);
stats_ris = build_stats_ris(out.ris);

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

fprintf('Skip MATLAB plotting in batch to avoid GUI transport issue; plots will be generated externally.\n');

json_power = struct();
json_power.run_id = 'paper_sweep_power_ga_rt_final';
json_power.seeds = seeds_full;
json_power.mc_per_seed = mc_per_seed;
json_power.n_total = N_total;
json_power.p_list = p_list;
json_power.alg_names = alg_names;
json_power.ga_rt_cfg = cfg_rt;
json_power.warm_start = warm_start;
json_power.chosen_cfg = best.cfg;
json_power.metrics = stats_power;
json_power.notes = 'Cold-start only; fair evaluate_system_rsma; independent RNG substreams; GA-RT realtime constrained.';
write_text_file(fullfile(res_dir, 'paper_sweep_power_ga_rt_final.json'), jsonencode(json_power));

json_ris = struct();
json_ris.run_id = 'ris_count_ga_rt_final';
json_ris.seeds = seeds_full;
json_ris.mc_per_seed = mc_per_seed;
json_ris.n_total = N_total;
json_ris.p_dbw = p_dbw_ris;
json_ris.ris_list = ris_list;
json_ris.alg_names = alg_names;
json_ris.ga_rt_cfg = cfg_rt;
json_ris.warm_start = warm_start;
json_ris.chosen_cfg = best.cfg;
json_ris.metrics = stats_ris;
json_ris.notes = 'Cold-start only; fair evaluate_system_rsma; independent RNG substreams; GA-RT realtime constrained.';
write_text_file(fullfile(res_dir, 'ris_count_ga_rt_final.json'), jsonencode(json_ris));

write_final_report(fullfile(res_dir, 'final_report.md'), best, out, p_list, ris_list, alg_names, N_total, cfg_rt);
fprintf('Saved final outputs.\n');
end

function cands = build_candidates_final()
base = struct( ...
    'ao_vio_bias_d', 0.20, ...
    'ao_vio_bias_s', 0.15, ...
    'ao_urgent_bias_mult', 1.8, ...
    'ao_accept_rel', 5e-3, ...
    'ao_accept_abs', 1e-4, ...
    'ao_max_outer_iter', 8, ...
    'ao_min_outer_iter', 3, ...
    'ao_wmmse_iter', 8, ...
    'ao_mm_iter', 5);

arr = [
    0.15 0.10 1.60 0.003 1e-4 8
    0.18 0.12 1.70 0.004 1e-4 8
    0.20 0.15 1.80 0.005 1e-4 8
    0.22 0.16 1.85 0.005 1e-4 9
    0.24 0.18 1.90 0.006 2e-4 9
    0.26 0.20 1.95 0.006 2e-4 10
    0.28 0.22 2.00 0.007 2e-4 10
    0.30 0.25 2.00 0.008 2e-4 10
    0.25 0.21 1.90 0.004 1e-4 10
    0.23 0.17 1.85 0.003 1e-4 9
    0.27 0.24 2.00 0.008 2e-4 9
    0.21 0.14 1.75 0.004 1e-4 8
    0.18 0.25 2.00 0.006 2e-4 10
    0.22 0.25 2.00 0.006 2e-4 10
    0.25 0.25 2.00 0.007 2e-4 10
    0.30 0.25 1.80 0.006 2e-4 9
    0.15 0.25 1.90 0.004 1e-4 9
    ];

cands = repmat(struct('cfg', base), 1, size(arr, 1));
for i = 1:size(arr, 1)
    cfgi = base;
    cfgi.ao_vio_bias_d = arr(i, 1);
    cfgi.ao_vio_bias_s = arr(i, 2);
    cfgi.ao_urgent_bias_mult = arr(i, 3);
    cfgi.ao_accept_rel = arr(i, 4);
    cfgi.ao_accept_abs = arr(i, 5);
    cfgi.ao_max_outer_iter = arr(i, 6);
    cands(i).cfg = cfgi;
end
end

function ev = evaluate_candidate(power, ris, alg_names, p_list, ris_list)
id_p = find(strcmpi(alg_names, 'proposed'), 1);
id_rt = find(strcmpi(alg_names, 'ga_rt'), 1);

num_p = numel(p_list);
num_r = numel(ris_list);
dq = zeros(num_p + num_r, 1);
dd = zeros(num_p + num_r, 1);
ds = zeros(num_p + num_r, 1);

for i = 1:num_p
    dq(i) = mean(power.urgent_qoe(:, i, id_p) - power.urgent_qoe(:, i, id_rt));
    dd(i) = mean(power.urgent_delay_vio(:, i, id_p) - power.urgent_delay_vio(:, i, id_rt));
    ds(i) = mean(power.urgent_semantic_vio(:, i, id_p) - power.urgent_semantic_vio(:, i, id_rt));
end
for i = 1:num_r
    j = num_p + i;
    dq(j) = mean(ris.urgent_qoe(:, i, id_p) - ris.urgent_qoe(:, i, id_rt));
    dd(j) = mean(ris.urgent_delay_vio(:, i, id_p) - ris.urgent_delay_vio(:, i, id_rt));
    ds(j) = mean(ris.urgent_semantic_vio(:, i, id_p) - ris.urgent_semantic_vio(:, i, id_rt));
end

pass_delay = all(dd <= 0.02 + 1e-12);
pass_main = mean(dq) < 0 && mean(ds) < 0;
pass = pass_delay && pass_main;

% Higher score is better.
score = -mean(dq) - mean(ds) - 5 * max(0, max(dd) - 0.02);

ev = struct();
ev.pass = pass;
ev.score = score;
ev.mean_delta_qoe = mean(dq);
ev.mean_delta_sem = mean(ds);
ev.mean_delta_delay = mean(dd);
ev.max_delta_delay = max(dd);
ev.delay_fail_count = sum(dd > 0.02);
end

function out = run_core(cfg0, cfg_override, seeds, mc_each, p_list, ris_list, p_dbw_ris, alg_names, cfg_rt, collect_norm)
Np = numel(p_list);
Nr = numel(ris_list);
A = numel(alg_names);
N_total = numel(seeds) * mc_each;

power.avg_qoe = zeros(N_total, Np, A);
power.urgent_qoe = zeros(N_total, Np, A);
power.normal_qoe = zeros(N_total, Np, A);
power.sum_rate = zeros(N_total, Np, A);
power.urgent_sum_rate = zeros(N_total, Np, A);
power.urgent_avg_rate = zeros(N_total, Np, A);
power.urgent_delay_vio = zeros(N_total, Np, A);
power.urgent_semantic_vio = zeros(N_total, Np, A);
power.ris_count = zeros(N_total, Np, A);
power.proposed_eval_calls = nan(N_total, Np);
power.ga_rt_eval_calls = nan(N_total, Np);
power.ga_rt_budget_target = nan(N_total, Np);

ris.avg_qoe = zeros(N_total, Nr, A);
ris.urgent_qoe = zeros(N_total, Nr, A);
ris.sum_rate = zeros(N_total, Nr, A);
ris.urgent_sum_rate = zeros(N_total, Nr, A);
ris.urgent_avg_rate = zeros(N_total, Nr, A);
ris.urgent_delay_vio = zeros(N_total, Nr, A);
ris.urgent_semantic_vio = zeros(N_total, Nr, A);
ris.ris_count = zeros(N_total, Nr, A);
ris.proposed_eval_calls = nan(N_total, Nr);
ris.ga_rt_eval_calls = nan(N_total, Nr);
ris.ga_rt_budget_target = nan(N_total, Nr);

norm_ev_power = init_norm_evidence(Np);
norm_ev_ris = init_norm_evidence(Nr);

fprintf('Run core: seeds=%s, mc_each=%d, total=%d\n', mat2str(seeds), mc_each, N_total);
row = 0;
for is = 1:numel(seeds)
    seed_base = seeds(is);
    for t = 1:mc_each
        row = row + 1;
        trial_seed = seed_base + t;

        cfg = apply_cfg_overrides_local(cfg0, cfg_override);
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        profile = build_profile_urgent_normal(cfg, geom, struct());
        [urgent_idx, normal_idx] = get_group_indices(cfg, profile);
        eval_opts = struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table);

        for ip = 1:Np
            p_dbw = p_list(ip);
            proposed_budget = [];
            for ia = 1:A
                alg = alg_names{ia};
                rng(derive_algo_seed(seed_base, t, ip, ia, 11), 'twister');
                [sol, eval_calls, budget_target, norm_rec] = solve_alg(cfg, ch, geom, profile, p_dbw, alg, proposed_budget, cfg_rt);
                if strcmpi(alg, 'proposed')
                    proposed_budget = eval_calls;
                    power.proposed_eval_calls(row, ip) = eval_calls;
                end
                if strcmpi(alg, 'ga_rt')
                    power.ga_rt_eval_calls(row, ip) = eval_calls;
                    power.ga_rt_budget_target(row, ip) = budget_target;
                end

                out_eval = evaluate_system_rsma(cfg, ch, geom, sol, profile, eval_opts);
                rate_vec_bps = get_rate_vec_bps_compat(out_eval, cfg.num_users); % bps
                urgent_sum_rate_bps = sum(rate_vec_bps(urgent_idx)); % bps
                urgent_avg_rate_bps = mean(rate_vec_bps(urgent_idx)); % bps
                power.avg_qoe(row, ip, ia) = out_eval.avg_qoe;
                power.urgent_qoe(row, ip, ia) = mean(out_eval.qoe_vec(urgent_idx));
                power.normal_qoe(row, ip, ia) = mean(out_eval.qoe_vec(normal_idx));
                power.sum_rate(row, ip, ia) = out_eval.sum_rate_bps;
                power.urgent_sum_rate(row, ip, ia) = urgent_sum_rate_bps;
                power.urgent_avg_rate(row, ip, ia) = urgent_avg_rate_bps;
                power.urgent_delay_vio(row, ip, ia) = mean(out_eval.delay_vio_vec(urgent_idx));
                power.urgent_semantic_vio(row, ip, ia) = mean(out_eval.semantic_vio_vec(urgent_idx));
                power.ris_count(row, ip, ia) = sum(sol.assign(:) > 0);

                if collect_norm && strcmpi(alg, 'norm')
                    norm_ev_power = append_norm_evidence(norm_ev_power, ip, norm_rec.ratio_d_raw, norm_rec.ratio_s_raw, cfg.max_ratio_d, cfg.max_ratio_s);
                end
            end
        end

        for ir = 1:Nr
            cfg2 = apply_cfg_overrides_local(cfg0, cfg_override);
            cfg2.num_ris = ris_list(ir);
            cfg2.ris_per_cell = ris_list(ir);
            geom2 = geometry(cfg2, trial_seed);
            ch2 = channel(cfg2, geom2, trial_seed);
            profile2 = build_profile_urgent_normal(cfg2, geom2, struct());
            [urgent_idx2, ~] = get_group_indices(cfg2, profile2);
            eval_opts2 = struct('semantic_mode', cfg2.semantic_mode, 'table_path', cfg2.semantic_table);

            proposed_budget2 = [];
            for ia = 1:A
                alg = alg_names{ia};
                rng(derive_algo_seed(seed_base, t, ir, ia, 22), 'twister');
                [sol2, eval_calls2, budget_target2, norm_rec2] = solve_alg(cfg2, ch2, geom2, profile2, p_dbw_ris, alg, proposed_budget2, cfg_rt);
                if strcmpi(alg, 'proposed')
                    proposed_budget2 = eval_calls2;
                    ris.proposed_eval_calls(row, ir) = eval_calls2;
                end
                if strcmpi(alg, 'ga_rt')
                    ris.ga_rt_eval_calls(row, ir) = eval_calls2;
                    ris.ga_rt_budget_target(row, ir) = budget_target2;
                end

                out_eval2 = evaluate_system_rsma(cfg2, ch2, geom2, sol2, profile2, eval_opts2);
                rate_vec_bps2 = get_rate_vec_bps_compat(out_eval2, cfg2.num_users); % bps
                urgent_sum_rate_bps2 = sum(rate_vec_bps2(urgent_idx2)); % bps
                urgent_avg_rate_bps2 = mean(rate_vec_bps2(urgent_idx2)); % bps
                ris.urgent_qoe(row, ir, ia) = mean(out_eval2.qoe_vec(urgent_idx2));
                ris.avg_qoe(row, ir, ia) = out_eval2.avg_qoe;
                ris.sum_rate(row, ir, ia) = out_eval2.sum_rate_bps;
                ris.urgent_sum_rate(row, ir, ia) = urgent_sum_rate_bps2;
                ris.urgent_avg_rate(row, ir, ia) = urgent_avg_rate_bps2;
                ris.urgent_delay_vio(row, ir, ia) = mean(out_eval2.delay_vio_vec(urgent_idx2));
                ris.urgent_semantic_vio(row, ir, ia) = mean(out_eval2.semantic_vio_vec(urgent_idx2));
                ris.ris_count(row, ir, ia) = sum(sol2.assign(:) > 0);

                if collect_norm && strcmpi(alg, 'norm')
                    norm_ev_ris = append_norm_evidence(norm_ev_ris, ir, norm_rec2.ratio_d_raw, norm_rec2.ratio_s_raw, cfg2.max_ratio_d, cfg2.max_ratio_s);
                end
            end
        end

        if mod(row, 10) == 0 || row == N_total
            fprintf('  sample %d/%d\n', row, N_total);
        end
    end
end

out = struct();
out.power = power;
out.ris = ris;
out.norm_ev_power = norm_ev_power;
out.norm_ev_ris = norm_ev_ris;
end

function [sol, eval_calls, budget_target, norm_rec] = solve_alg(cfg, ch, geom, profile, p_dbw, alg, proposed_budget, cfg_rt)
norm_rec = struct('ratio_d_raw', [], 'ratio_s_raw', []);
budget_target = nan;
eval_calls = nan;

switch lower(alg)
    case 'proposed'
        [assign, theta_all, V, ao_log] = ua_qoe_ao(cfg, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, profile);
        sol = struct('assign', assign(:), 'theta_all', theta_all, 'V', V);
        eval_calls = ao_log.eval_calls;
    case 'random'
        assign = pick_random_capacity(cfg);
        [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign, profile, struct());
        sol = sol_fixed;
    case 'norm'
        assign = pick_norm_capacity(cfg, ch);
        [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign, profile, struct());
        sol = sol_fixed;
        out_eval = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table));
        norm_rec.ratio_d_raw = out_eval.T_tx ./ (profile.d_k(:) + cfg.eps);
        norm_rec.ratio_s_raw = out_eval.D ./ (profile.dmax_k(:) + cfg.eps);
    case 'ga_rt'
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'random_fixed';
        opts.pop_size = cfg_rt.pop_size;
        opts.num_generations = cfg_rt.num_generations;
        target = cfg_rt.budget_evals;
        if ~isempty(proposed_budget) && isfinite(proposed_budget) && proposed_budget > 0
            target = min(target, proposed_budget);
        end
        opts.budget_evals = target;
        [assign, ~, ~, info] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), cfg.num_ris, cfg.k0);
        sol = build_light_solution(cfg, ch, assign, p_dbw);
        eval_calls = info.eval_count;
        budget_target = target;
    otherwise
        error('Unsupported algorithm: %s', alg);
end
end

function out = build_stats_power(power)
out = struct();
[out.avg_qoe.mean, out.avg_qoe.ci95] = calc_mean_ci(power.avg_qoe);
[out.urgent_qoe.mean, out.urgent_qoe.ci95] = calc_mean_ci(power.urgent_qoe);
[out.normal_qoe.mean, out.normal_qoe.ci95] = calc_mean_ci(power.normal_qoe);
[out.sum_rate.mean, out.sum_rate.ci95] = calc_mean_ci(power.sum_rate);
[out.urgent_sum_rate.mean, out.urgent_sum_rate.ci95] = calc_mean_ci(power.urgent_sum_rate);
[out.urgent_avg_rate.mean, out.urgent_avg_rate.ci95] = calc_mean_ci(power.urgent_avg_rate);
[out.urgent_delay_vio.mean, out.urgent_delay_vio.ci95] = calc_mean_ci(power.urgent_delay_vio);
[out.urgent_semantic_vio.mean, out.urgent_semantic_vio.ci95] = calc_mean_ci(power.urgent_semantic_vio);
[out.ris_count.mean, out.ris_count.ci95] = calc_mean_ci(power.ris_count);
[out.proposed_eval_calls.mean, out.proposed_eval_calls.ci95] = calc_mean_ci2d(power.proposed_eval_calls);
[out.ga_rt_eval_calls.mean, out.ga_rt_eval_calls.ci95] = calc_mean_ci2d(power.ga_rt_eval_calls);
[out.ga_rt_budget_target.mean, out.ga_rt_budget_target.ci95] = calc_mean_ci2d(power.ga_rt_budget_target);
out = ensure_urgent_rate_metrics(out);
end

function out = build_stats_ris(ris)
out = struct();
[out.avg_qoe.mean, out.avg_qoe.ci95] = calc_mean_ci(ris.avg_qoe);
[out.urgent_qoe.mean, out.urgent_qoe.ci95] = calc_mean_ci(ris.urgent_qoe);
[out.sum_rate.mean, out.sum_rate.ci95] = calc_mean_ci(ris.sum_rate);
[out.urgent_sum_rate.mean, out.urgent_sum_rate.ci95] = calc_mean_ci(ris.urgent_sum_rate);
[out.urgent_avg_rate.mean, out.urgent_avg_rate.ci95] = calc_mean_ci(ris.urgent_avg_rate);
[out.urgent_delay_vio.mean, out.urgent_delay_vio.ci95] = calc_mean_ci(ris.urgent_delay_vio);
[out.urgent_semantic_vio.mean, out.urgent_semantic_vio.ci95] = calc_mean_ci(ris.urgent_semantic_vio);
[out.ris_count.mean, out.ris_count.ci95] = calc_mean_ci(ris.ris_count);
[out.proposed_eval_calls.mean, out.proposed_eval_calls.ci95] = calc_mean_ci2d(ris.proposed_eval_calls);
[out.ga_rt_eval_calls.mean, out.ga_rt_eval_calls.ci95] = calc_mean_ci2d(ris.ga_rt_eval_calls);
[out.ga_rt_budget_target.mean, out.ga_rt_budget_target.ci95] = calc_mean_ci2d(ris.ga_rt_budget_target);
out = ensure_urgent_rate_metrics(out);
end

function write_final_report(path_md, best, out, p_list, ris_list, alg_names, N_total, cfg_rt)
id_p = find(strcmpi(alg_names, 'proposed'), 1);
id_rt = find(strcmpi(alg_names, 'ga_rt'), 1);

fid = fopen(path_md, 'w');
cleanup_obj = onCleanup(@() fclose(fid));

fprintf(fid, '# Final Report (GA-RT, Cold-Start)\n\n');
fprintf(fid, '- warm_start: false\n');
fprintf(fid, '- seeds: [40, 41, 42, 43, 44], mc_per_seed: 30, N_total: %d\n', N_total);
fprintf(fid, '- GA-RT: pop=%d, gen=%d, budget=%d, light_solve_only=true\n\n', cfg_rt.pop_size, cfg_rt.num_generations, cfg_rt.budget_evals);

fprintf(fid, '## Chosen Parameters\n');
fprintf(fid, '| key | value |\n|---|---:|\n');
fprintf(fid, '| ao_vio_bias_d | %.4f |\n', best.cfg.ao_vio_bias_d);
fprintf(fid, '| ao_vio_bias_s | %.4f |\n', best.cfg.ao_vio_bias_s);
fprintf(fid, '| ao_urgent_bias_mult | %.4f |\n', best.cfg.ao_urgent_bias_mult);
fprintf(fid, '| ao_accept_rel | %.6f |\n', best.cfg.ao_accept_rel);
fprintf(fid, '| ao_accept_abs | %.6f |\n', best.cfg.ao_accept_abs);
fprintf(fid, '| ao_max_outer_iter | %d |\n', best.cfg.ao_max_outer_iter);
fprintf(fid, '| ao_min_outer_iter | %d |\n', best.cfg.ao_min_outer_iter);
fprintf(fid, '| ao_wmmse_iter | %d |\n', best.cfg.ao_wmmse_iter);
fprintf(fid, '| ao_mm_iter | %d |\n\n', best.cfg.ao_mm_iter);

fprintf(fid, '## 12-point Delta Table (proposed - GA-RT)\n');
fprintf(fid, '| point | d_urgent_qoe_cost | d_urgent_semantic_vio | d_urgent_delay_vio | d_sum_rate_Mbps |\n');
fprintf(fid, '|---|---:|---:|---:|---:|\n');
for i = 1:numel(p_list)
    duq = mean(out.power.urgent_qoe(:, i, id_p) - out.power.urgent_qoe(:, i, id_rt));
    dsv = mean(out.power.urgent_semantic_vio(:, i, id_p) - out.power.urgent_semantic_vio(:, i, id_rt));
    ddv = mean(out.power.urgent_delay_vio(:, i, id_p) - out.power.urgent_delay_vio(:, i, id_rt));
    dsr = mean((out.power.sum_rate(:, i, id_p) - out.power.sum_rate(:, i, id_rt)) / 1e6);
    fprintf(fid, '| power %.2f | %.6f | %.6f | %.6f | %.6f |\n', p_list(i), duq, dsv, ddv, dsr);
end
for i = 1:numel(ris_list)
    duq = mean(out.ris.urgent_qoe(:, i, id_p) - out.ris.urgent_qoe(:, i, id_rt));
    dsv = mean(out.ris.urgent_semantic_vio(:, i, id_p) - out.ris.urgent_semantic_vio(:, i, id_rt));
    ddv = mean(out.ris.urgent_delay_vio(:, i, id_p) - out.ris.urgent_delay_vio(:, i, id_rt));
    dsr = mean((out.ris.sum_rate(:, i, id_p) - out.ris.sum_rate(:, i, id_rt)) / 1e6);
    fprintf(fid, '| ris %d | %.6f | %.6f | %.6f | %.6f |\n', ris_list(i), duq, dsv, ddv, dsr);
end
fprintf(fid, '\n');

fprintf(fid, '## Mean Delta and 95%% CI\n');
fprintf(fid, '| metric | mean_delta | ci95 |\n');
fprintf(fid, '|---|---:|---:|\n');
[mu_q, ci_q] = overall_delta_ci(out.power, out.ris, id_p, id_rt, 'urgent_qoe', N_total);
[mu_s, ci_s] = overall_delta_ci(out.power, out.ris, id_p, id_rt, 'urgent_semantic_vio', N_total);
[mu_d, ci_d] = overall_delta_ci(out.power, out.ris, id_p, id_rt, 'urgent_delay_vio', N_total);
fprintf(fid, '| urgent_qoe_cost | %.6f | %.6f |\n', mu_q, ci_q);
fprintf(fid, '| urgent_semantic_vio | %.6f | %.6f |\n', mu_s, ci_s);
fprintf(fid, '| urgent_delay_vio | %.6f | %.6f |\n\n', mu_d, ci_d);

fprintf(fid, '## Norm Saturation Evidence (ratio P50/P90 + clamp_hit_rate)\n');
fprintf(fid, '| point | ratio_d_p50 | ratio_d_p90 | ratio_s_p50 | ratio_s_p90 | clamp_hit_rate_d | clamp_hit_rate_s |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
for i = 1:numel(p_list)
    [d50, d90, s50, s90, hd, hs] = ev_stats(out.norm_ev_power, i);
    fprintf(fid, '| power %.2f | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |\n', p_list(i), d50, d90, s50, s90, hd, hs);
end
for i = 1:numel(ris_list)
    [d50, d90, s50, s90, hd, hs] = ev_stats(out.norm_ev_ris, i);
    fprintf(fid, '| ris %d | %.6f | %.6f | %.6f | %.6f | %.6f | %.6f |\n', ris_list(i), d50, d90, s50, s90, hd, hs);
end
fprintf(fid, '\n');

delay_guard_ok = max_delay_delta_12pt(out.power, out.ris, id_p, id_rt) <= 0.02 + 1e-12;
main_ok = (mu_q < 0) && (mu_s < 0);
if main_ok && delay_guard_ok
    fprintf(fid, 'Conclusion: under real-time GA-RT baseline, proposed is significantly better on urgent_qoe cost and urgent_semantic_vio, and delay_vio does not worsen.\n');
else
    fprintf(fid, 'Conclusion: current configuration does not satisfy all required conditions and needs further tuning.\n');
end

clear cleanup_obj;
end

function v = max_delay_delta_12pt(power, ris, id_p, id_rt)
d1 = mean(power.urgent_delay_vio(:, :, id_p) - power.urgent_delay_vio(:, :, id_rt), 1);
d2 = mean(ris.urgent_delay_vio(:, :, id_p) - ris.urgent_delay_vio(:, :, id_rt), 1);
v = max([d1(:); d2(:)]);
end

function [mu, ci] = overall_delta_ci(power, ris, id_p, id_rt, field_name, N_total)
switch lower(field_name)
    case 'urgent_qoe'
        A = power.urgent_qoe;
        B = ris.urgent_qoe;
    case 'urgent_delay_vio'
        A = power.urgent_delay_vio;
        B = ris.urgent_delay_vio;
    otherwise
        A = power.urgent_semantic_vio;
        B = ris.urgent_semantic_vio;
end
d1 = squeeze(mean(A(:, :, id_p) - A(:, :, id_rt), 2));
d2 = squeeze(mean(B(:, :, id_p) - B(:, :, id_rt), 2));
d = 0.5 * (d1 + d2);
mu = mean(d);
ci = 1.96 * std(d) / sqrt(N_total);
end

function s = init_norm_evidence(n)
s = struct();
s.ratio_d = cell(n, 1);
s.ratio_s = cell(n, 1);
s.hit_d = zeros(n, 1);
s.hit_s = zeros(n, 1);
s.count = zeros(n, 1);
end

function s = append_norm_evidence(s, idx, ratio_d_raw, ratio_s_raw, max_d, max_s)
ratio_d_raw = real(ratio_d_raw(:));
ratio_s_raw = real(ratio_s_raw(:));
s.ratio_d{idx} = [s.ratio_d{idx}; ratio_d_raw];
s.ratio_s{idx} = [s.ratio_s{idx}; ratio_s_raw];
s.hit_d(idx) = s.hit_d(idx) + sum(ratio_d_raw >= max_d - 1e-12);
s.hit_s(idx) = s.hit_s(idx) + sum(ratio_s_raw >= max_s - 1e-12);
s.count(idx) = s.count(idx) + numel(ratio_d_raw);
end

function [d50, d90, s50, s90, hd, hs] = ev_stats(ev, idx)
rd = ev.ratio_d{idx};
rs = ev.ratio_s{idx};
if isempty(rd), rd = 0; end
if isempty(rs), rs = 0; end
d50 = quantile(rd, 0.5);
d90 = quantile(rd, 0.9);
s50 = quantile(rs, 0.5);
s90 = quantile(rs, 0.9);
hd = ev.hit_d(idx) / max(1, ev.count(idx));
hs = ev.hit_s(idx) / max(1, ev.count(idx));
end

function [urgent_idx, normal_idx] = get_group_indices(cfg, profile)
K = cfg.num_users;
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('run_ga_rt_final:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required for unified grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
if ~isempty(intersect(urgent_idx, normal_idx))
    error('run_ga_rt_final:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx(:); normal_idx(:)]);
assert(numel(cover) == K && all(cover(:) == (1:K).'), ...
    'run_ga_rt_final:group_coverage: urgent/normal union must cover all users');
end

function out = derive_algo_seed(base_seed, trial_idx, x_idx, alg_idx, tag)
out = base_seed * 10000000 + trial_idx * 100000 + x_idx * 1000 + alg_idx * 10 + tag;
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
    if pick > 0, cap(pick) = cap(pick) - 1; end
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
    if best_gain(k) <= 0, assign(k) = 0; continue; end
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
    if ~placed, assign(k) = 0; end
end
end

function assign = enforce_capacity(assign, L, k0)
assign = assign(:);
for l = 1:L
    idx = find(assign == l);
    if numel(idx) > k0
        assign(idx((k0 + 1):end)) = 0;
    end
end
end

function sol = build_light_solution(cfg, ch, assign, p_dbw)
theta_all = ch.theta;
h_eff = effective_channel(cfg, ch, assign, theta_all);
[V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 3);
sol = struct('assign', assign(:), 'theta_all', theta_all, 'V', V);
end

function cfg = apply_cfg_overrides_local(cfg, overrides)
if nargin < 2 || isempty(overrides), return; end
f = fieldnames(overrides);
for i = 1:numel(f)
    cfg.(f{i}) = overrides.(f{i});
end
end

function [mu, ci95] = calc_mean_ci(x)
N = size(x, 1);
nx = size(x, 2);
na = size(x, 3);
mu = reshape(mean(x, 1), [nx, na]);
sd = reshape(std(x, 0, 1), [nx, na]);
ci95 = 1.96 * sd / sqrt(N);
end

function [mu, ci95] = calc_mean_ci2d(x)
N = size(x, 1);
mu = mean(x, 1, 'omitnan');
sd = std(x, 0, 1, 'omitnan');
ci95 = 1.96 * sd / sqrt(N);
mu = mu(:).';
ci95 = ci95(:).';
end

function plot_mean_lines(x, y_mean, alg_names, out_path, x_label, y_label, ttl)
fig = figure('Color', 'w', 'Visible', 'off');
hold on;
colors = lines(numel(alg_names));
markers = {'o','s','d','^','v','x'};
for a = 1:numel(alg_names)
    plot(x(:).', y_mean(:, a).', ['-' markers{min(a, numel(markers))}], ...
        'Color', colors(a, :), 'LineWidth', 1.6, 'MarkerSize', 5, ...
        'DisplayName', legend_label(alg_names{a}));
end
grid on;
xlabel(x_label);
ylabel(y_label);
title(ttl);
legend('Location', 'best');
saveas(fig, out_path);
close(fig);
end

function lbl = legend_label(name)
switch lower(name)
    case 'ga_rt'
        lbl = 'GA-RT';
    otherwise
        lbl = name;
end
end

function write_text_file(path_name, txt)
fid = fopen(path_name, 'w');
if fid < 0, error('Cannot open file for writing: %s', path_name); end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', txt);
clear cleanup_obj;
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
    error('run_ga_rt_final:missing_rate_vec', 'Cannot recover per-user rate vector.');
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
    stats.urgent_sum_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
if ~isfield(stats, 'urgent_avg_rate')
    stats.urgent_avg_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
end
