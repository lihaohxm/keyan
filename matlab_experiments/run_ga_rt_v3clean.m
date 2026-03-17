function run_ga_rt_v3clean()
%RUN_GA_RT_V3CLEAN Multi-seed clean run with GA-RT real-time baseline.
% Outputs:
%   results/paper_sweep_power_ga_rt_v3clean.json
%   results/ris_count_ga_rt_v3clean.json
%   results/ga_rt_v3clean_report.md

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg0 = config();
seeds = [40 41 42];
mc_each = 30;
p_list = cfg0.p_dbw_list(:).';
ris_list = [1 2 3 4];
p_dbw_ris = -8;

alg_names = {'proposed', 'random', 'norm', 'ga_rt', 'ga_ub'};
A = numel(alg_names);
Np = numel(p_list);
Nr = numel(ris_list);
N_total = numel(seeds) * mc_each;

cfg_rt = struct('pop_size', 8, 'num_generations', 6, 'budget_evals', 40, 'light_solve_only', true);

power.avg_qoe = zeros(N_total, Np, A);
power.urgent_qoe = zeros(N_total, Np, A);
power.normal_qoe = zeros(N_total, Np, A);
power.sum_rate = zeros(N_total, Np, A);
power.urgent_delay_vio = zeros(N_total, Np, A);
power.urgent_semantic_vio = zeros(N_total, Np, A);
power.ris_count = zeros(N_total, Np, A);
power.proposed_eval_calls = nan(N_total, Np);
power.ga_rt_eval_calls = nan(N_total, Np);
power.ga_rt_budget_target = nan(N_total, Np);

ris.avg_qoe = zeros(N_total, Nr, A);
ris.urgent_qoe = zeros(N_total, Nr, A);
ris.sum_rate = zeros(N_total, Nr, A);
ris.urgent_delay_vio = zeros(N_total, Nr, A);
ris.urgent_semantic_vio = zeros(N_total, Nr, A);
ris.ris_count = zeros(N_total, Nr, A);
ris.proposed_eval_calls = nan(N_total, Nr);
ris.ga_rt_eval_calls = nan(N_total, Nr);
ris.ga_rt_budget_target = nan(N_total, Nr);

norm_ev_power = init_norm_evidence(Np);
norm_ev_ris = init_norm_evidence(Nr);

fprintf('=== v3clean: seeds=%s, mc_each=%d ===\n', mat2str(seeds), mc_each);
row = 0;
for is = 1:numel(seeds)
    seed_base = seeds(is);
    for t = 1:mc_each
        row = row + 1;
        trial_seed = seed_base + t;
        rng(trial_seed, 'twister');

        % ---------- Power sweep with warm-start ----------
        cfg = cfg0;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        profile = build_profile_urgent_normal(cfg, geom, struct());
        [urgent_idx, normal_idx] = get_group_indices(cfg, profile);
        eval_opts = struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table);

        ws = struct();
        ws.proposed = [];
        ws.random = [];
        ws.norm = [];
        ws.ga_rt = [];
        ws.ga_ub = [];

        for ip = 1:Np
            p_dbw = p_list(ip);
            proposed_budget = [];
            for ia = 1:A
                alg = alg_names{ia};
                rng(derive_algo_seed(seed_base, t, ip, ia), 'twister');
                [sol, eval_calls, budget_target, norm_rec, assign_ws] = solve_alg_power(cfg, ch, geom, profile, p_dbw, alg, ws.(alg), proposed_budget);
                ws.(alg) = assign_ws;
                if strcmpi(alg, 'proposed')
                    proposed_budget = eval_calls;
                    power.proposed_eval_calls(row, ip) = eval_calls;
                end
                if strcmpi(alg, 'ga_rt')
                    power.ga_rt_eval_calls(row, ip) = eval_calls;
                    power.ga_rt_budget_target(row, ip) = budget_target;
                end

                out = evaluate_system_rsma(cfg, ch, geom, sol, profile, eval_opts);
                power.avg_qoe(row, ip, ia) = out.avg_qoe;
                power.urgent_qoe(row, ip, ia) = mean(out.qoe_vec(urgent_idx));
                power.normal_qoe(row, ip, ia) = mean(out.qoe_vec(normal_idx));
                power.sum_rate(row, ip, ia) = out.sum_rate_bps;
                power.urgent_delay_vio(row, ip, ia) = mean(out.delay_vio_vec(urgent_idx));
                power.urgent_semantic_vio(row, ip, ia) = mean(out.semantic_vio_vec(urgent_idx));
                power.ris_count(row, ip, ia) = sum(sol.assign(:) > 0);

                if strcmpi(alg, 'norm')
                    norm_ev_power = append_norm_evidence(norm_ev_power, ip, norm_rec.ratio_d_raw, norm_rec.ratio_s_raw, cfg.max_ratio_d, cfg.max_ratio_s);
                end
            end
        end

        % ---------- RIS sweep ----------
        for ir = 1:Nr
            cfg2 = cfg0;
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
                rng(derive_algo_seed(seed_base, t, ir, ia), 'twister');
                [sol2, eval_calls2, budget_target2, norm_rec2, ~] = solve_alg_ris(cfg2, ch2, geom2, profile2, p_dbw_ris, alg, [], proposed_budget2);
                if strcmpi(alg, 'proposed')
                    proposed_budget2 = eval_calls2;
                    ris.proposed_eval_calls(row, ir) = eval_calls2;
                end
                if strcmpi(alg, 'ga_rt')
                    ris.ga_rt_eval_calls(row, ir) = eval_calls2;
                    ris.ga_rt_budget_target(row, ir) = budget_target2;
                end

                out2 = evaluate_system_rsma(cfg2, ch2, geom2, sol2, profile2, eval_opts2);
                ris.urgent_qoe(row, ir, ia) = mean(out2.qoe_vec(urgent_idx2));
                ris.avg_qoe(row, ir, ia) = out2.avg_qoe;
                ris.sum_rate(row, ir, ia) = out2.sum_rate_bps;
                ris.urgent_delay_vio(row, ir, ia) = mean(out2.delay_vio_vec(urgent_idx2));
                ris.urgent_semantic_vio(row, ir, ia) = mean(out2.semantic_vio_vec(urgent_idx2));
                ris.ris_count(row, ir, ia) = sum(sol2.assign(:) > 0);

                if strcmpi(alg, 'norm')
                    norm_ev_ris = append_norm_evidence(norm_ev_ris, ir, norm_rec2.ratio_d_raw, norm_rec2.ratio_s_raw, cfg2.max_ratio_d, cfg2.max_ratio_s);
                end
            end
        end

        if mod(row, 10) == 0 || row == N_total
            fprintf('  sample %d/%d\n', row, N_total);
        end
    end
end

stats_power = build_stats_power(power);
stats_ris = build_stats_ris(ris);

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

plot_mean_lines(p_list, stats_power.urgent_qoe.mean, alg_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_v3clean_urgent_qoe.png'), 'Urgent QoE Cost', 'Urgent QoE Cost vs p');
plot_mean_lines(p_list, stats_power.urgent_delay_vio.mean, alg_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_v3clean_urgent_delay_vio.png'), 'Urgent Delay Violation', 'Urgent Delay Violation vs p');
plot_mean_lines(p_list, stats_power.urgent_semantic_vio.mean, alg_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_v3clean_urgent_semantic_vio.png'), 'Urgent Semantic Violation', 'Urgent Semantic Violation vs p');
plot_mean_lines(ris_list, stats_ris.urgent_qoe.mean, alg_names, fullfile(fig_dir, 'ris_count_ga_rt_v3clean_urgent_qoe.png'), 'Urgent QoE Cost', 'Urgent QoE Cost vs RIS');

json_power = struct();
json_power.run_id = 'paper_sweep_power_ga_rt_v3clean';
json_power.seeds = seeds;
json_power.mc_each = mc_each;
json_power.n_total = N_total;
json_power.p_list = p_list;
json_power.alg_names = alg_names;
json_power.ga_rt_cfg = cfg_rt;
json_power.warm_start = true;
json_power.metrics = stats_power;
json_power.notes = 'v3clean multi-seed aggregate; warm-start assignment enabled along p sweep.';
write_text_file(fullfile(res_dir, 'paper_sweep_power_ga_rt_v3clean.json'), jsonencode(json_power));

json_ris = struct();
json_ris.run_id = 'ris_count_ga_rt_v3clean';
json_ris.seeds = seeds;
json_ris.mc_each = mc_each;
json_ris.n_total = N_total;
json_ris.p_dbw = p_dbw_ris;
json_ris.ris_list = ris_list;
json_ris.alg_names = alg_names;
json_ris.ga_rt_cfg = cfg_rt;
json_ris.metrics = stats_ris;
json_ris.notes = 'v3clean multi-seed aggregate.';
write_text_file(fullfile(res_dir, 'ris_count_ga_rt_v3clean.json'), jsonencode(json_ris));

write_report_v3clean(fullfile(res_dir, 'ga_rt_v3clean_report.md'), p_list, ris_list, alg_names, power, ris, norm_ev_power, norm_ev_ris, N_total);
fprintf('Saved v3clean outputs.\n');
end

function [sol, eval_calls, budget_target, norm_rec, assign_ws] = solve_alg_power(cfg, ch, geom, profile, p_dbw, alg, warm_assign, proposed_budget)
norm_rec = struct('ratio_d_raw', [], 'ratio_s_raw', []);
budget_target = nan;
eval_calls = nan;
assign_ws = [];

switch lower(alg)
    case 'proposed'
        cfgp = cfg;
        if ~isempty(warm_assign) && numel(warm_assign) == cfg.num_users
            cfgp.ao_init_assign = warm_assign(:);
        end
        [assign, theta_all, V, ao_log] = ua_qoe_ao(cfgp, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, profile);
        sol = struct('assign', assign(:), 'theta_all', theta_all, 'V', V);
        eval_calls = ao_log.eval_calls;
        assign_ws = assign(:);
    case 'random'
        assign = pick_random_capacity(cfg);
        [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign, profile, struct());
        sol = sol_fixed;
        assign_ws = assign(:);
    case 'norm'
        assign = pick_norm_capacity(cfg, ch);
        [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign, profile, struct());
        sol = sol_fixed;
        assign_ws = assign(:);
        out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table));
        norm_rec.ratio_d_raw = out.T_tx ./ (profile.d_k(:) + cfg.eps);
        norm_rec.ratio_s_raw = out.D ./ (profile.dmax_k(:) + cfg.eps);
    case 'ga_rt'
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'align_fixed';
        opts.pop_size = 8;
        opts.num_generations = 6;
        target = 40;
        if ~isempty(proposed_budget) && isfinite(proposed_budget) && proposed_budget > 0
            target = min(target, proposed_budget);
        end
        opts.budget_evals = target;
        if ~isempty(warm_assign)
            opts.seed_assignment = warm_assign(:);
        end
        [assign, ~, ~, info] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), cfg.num_ris, cfg.k0);
        sol = build_light_solution(cfg, ch, assign, p_dbw);
        eval_calls = info.eval_count;
        budget_target = target;
        assign_ws = assign(:);
    otherwise % ga_ub
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'align_fixed';
        opts.pop_size = cfg.ga_Np;
        opts.num_generations = cfg.ga_Niter;
        if ~isempty(warm_assign)
            opts.seed_assignment = warm_assign(:);
        end
        [assign, ~, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), cfg.num_ris, cfg.k0);
        [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign, profile, struct());
        sol = sol_fixed;
        assign_ws = assign(:);
end
end

function [sol, eval_calls, budget_target, norm_rec, assign_ws] = solve_alg_ris(cfg, ch, geom, profile, p_dbw, alg, warm_assign, proposed_budget)
[sol, eval_calls, budget_target, norm_rec, assign_ws] = solve_alg_power(cfg, ch, geom, profile, p_dbw, alg, warm_assign, proposed_budget);
end

function out = build_stats_power(power)
out = struct();
[out.avg_qoe.mean, out.avg_qoe.ci95] = calc_mean_ci(power.avg_qoe);
[out.urgent_qoe.mean, out.urgent_qoe.ci95] = calc_mean_ci(power.urgent_qoe);
[out.normal_qoe.mean, out.normal_qoe.ci95] = calc_mean_ci(power.normal_qoe);
[out.sum_rate.mean, out.sum_rate.ci95] = calc_mean_ci(power.sum_rate);
[out.urgent_delay_vio.mean, out.urgent_delay_vio.ci95] = calc_mean_ci(power.urgent_delay_vio);
[out.urgent_semantic_vio.mean, out.urgent_semantic_vio.ci95] = calc_mean_ci(power.urgent_semantic_vio);
[out.ris_count.mean, out.ris_count.ci95] = calc_mean_ci(power.ris_count);
[out.proposed_eval_calls.mean, out.proposed_eval_calls.ci95] = calc_mean_ci2d(power.proposed_eval_calls);
[out.ga_rt_eval_calls.mean, out.ga_rt_eval_calls.ci95] = calc_mean_ci2d(power.ga_rt_eval_calls);
[out.ga_rt_budget_target.mean, out.ga_rt_budget_target.ci95] = calc_mean_ci2d(power.ga_rt_budget_target);
end

function out = build_stats_ris(ris)
out = struct();
[out.avg_qoe.mean, out.avg_qoe.ci95] = calc_mean_ci(ris.avg_qoe);
[out.urgent_qoe.mean, out.urgent_qoe.ci95] = calc_mean_ci(ris.urgent_qoe);
[out.sum_rate.mean, out.sum_rate.ci95] = calc_mean_ci(ris.sum_rate);
[out.urgent_delay_vio.mean, out.urgent_delay_vio.ci95] = calc_mean_ci(ris.urgent_delay_vio);
[out.urgent_semantic_vio.mean, out.urgent_semantic_vio.ci95] = calc_mean_ci(ris.urgent_semantic_vio);
[out.ris_count.mean, out.ris_count.ci95] = calc_mean_ci(ris.ris_count);
[out.proposed_eval_calls.mean, out.proposed_eval_calls.ci95] = calc_mean_ci2d(ris.proposed_eval_calls);
[out.ga_rt_eval_calls.mean, out.ga_rt_eval_calls.ci95] = calc_mean_ci2d(ris.ga_rt_eval_calls);
[out.ga_rt_budget_target.mean, out.ga_rt_budget_target.ci95] = calc_mean_ci2d(ris.ga_rt_budget_target);
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

function write_report_v3clean(path_md, p_list, ris_list, alg_names, power, ris, norm_ev_power, norm_ev_ris, N_total)
id_p = find(strcmpi(alg_names, 'proposed'), 1);
id_rt = find(strcmpi(alg_names, 'ga_rt'), 1);

fid = fopen(path_md, 'w');
cleanup_obj = onCleanup(@() fclose(fid));

fprintf(fid, '# GA-RT v3clean Report\n\n');
fprintf(fid, '## Table 1: 12-point Delta (proposed - GA-RT)\n');
fprintf(fid, '| point | d_urgent_qoe_cost | d_urgent_delay_vio | d_urgent_semantic_vio | d_sum_rate_Mbps | d_ris_count |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|\n');
for i = 1:numel(p_list)
    duq = mean(power.urgent_qoe(:, i, id_p) - power.urgent_qoe(:, i, id_rt));
    ddv = mean(power.urgent_delay_vio(:, i, id_p) - power.urgent_delay_vio(:, i, id_rt));
    dsv = mean(power.urgent_semantic_vio(:, i, id_p) - power.urgent_semantic_vio(:, i, id_rt));
    dsr = mean((power.sum_rate(:, i, id_p) - power.sum_rate(:, i, id_rt)) / 1e6);
    drc = mean(power.ris_count(:, i, id_p) - power.ris_count(:, i, id_rt));
    fprintf(fid, '| power %.2f | %.4f | %.4f | %.4f | %.4f | %.4f |\n', p_list(i), duq, ddv, dsv, dsr, drc);
end
for i = 1:numel(ris_list)
    duq = mean(ris.urgent_qoe(:, i, id_p) - ris.urgent_qoe(:, i, id_rt));
    ddv = mean(ris.urgent_delay_vio(:, i, id_p) - ris.urgent_delay_vio(:, i, id_rt));
    dsv = mean(ris.urgent_semantic_vio(:, i, id_p) - ris.urgent_semantic_vio(:, i, id_rt));
    dsr = mean((ris.sum_rate(:, i, id_p) - ris.sum_rate(:, i, id_rt)) / 1e6);
    drc = mean(ris.ris_count(:, i, id_p) - ris.ris_count(:, i, id_rt));
    fprintf(fid, '| ris %d | %.4f | %.4f | %.4f | %.4f | %.4f |\n', ris_list(i), duq, ddv, dsv, dsr, drc);
end
fprintf(fid, '\n');

fprintf(fid, '## Table 2: Mean Delta + 95%% CI (N_total=%d, CI=1.96*std/sqrt(N_total))\n', N_total);
fprintf(fid, '| metric | mean_delta | ci95 |\n');
fprintf(fid, '|---|---:|---:|\n');
[m1, c1] = overall_delta_ci(power, ris, id_p, id_rt, 'urgent_qoe', N_total);
[m2, c2] = overall_delta_ci(power, ris, id_p, id_rt, 'urgent_semantic_vio', N_total);
[m3, c3] = overall_delta_ci(power, ris, id_p, id_rt, 'urgent_delay_vio', N_total);
fprintf(fid, '| urgent_qoe_cost | %.6f | %.6f |\n', m1, c1);
fprintf(fid, '| urgent_semantic_vio | %.6f | %.6f |\n', m2, c2);
fprintf(fid, '| urgent_delay_vio | %.6f | %.6f |\n', m3, c3);
fprintf(fid, '\n');

fprintf(fid, '## Table 3: Norm Saturation Evidence\n');
fprintf(fid, '| point | ratio_d_p50 | ratio_d_p90 | ratio_s_p50 | ratio_s_p90 | clamp_hit_rate_d | clamp_hit_rate_s |\n');
fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|\n');
for i = 1:numel(p_list)
    [d50, d90, s50, s90, hd, hs] = ev_stats(norm_ev_power, i);
    fprintf(fid, '| power %.2f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n', p_list(i), d50, d90, s50, s90, hd, hs);
end
for i = 1:numel(ris_list)
    [d50, d90, s50, s90, hd, hs] = ev_stats(norm_ev_ris, i);
    fprintf(fid, '| ris %d | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n', ris_list(i), d50, d90, s50, s90, hd, hs);
end
fprintf(fid, '\n');
fprintf(fid, 'If ratio P50 > 1 and clamp_hit_rate is near 100%%, violation near 1 indicates saturation behavior rather than evaluation bug.\n');
clear cleanup_obj;
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

function [urgent_idx, normal_idx] = get_group_indices(cfg, profile)
K = cfg.num_users;
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('run_ga_rt_v3clean:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required for unified grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
if ~isempty(intersect(urgent_idx, normal_idx))
    error('run_ga_rt_v3clean:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx(:); normal_idx(:)]);
assert(numel(cover) == K && all(cover(:) == (1:K).'), ...
    'run_ga_rt_v3clean:group_coverage: urgent/normal union must cover all users');
end

function out = derive_algo_seed(base_seed, trial_idx, p_idx, alg_idx)
out = base_seed + trial_idx * 1000000 + p_idx * 10000 + alg_idx * 100;
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

function plot_mean_lines(x, y_mean, alg_names, out_path, y_label, ttl)
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
xlabel('x');
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
    case 'ga_ub'
        lbl = 'GA-UB';
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
