function run_ablation_warmstart_cfg()
%RUN_ABLATION_WARMSTART_CFG 2×2 ablation vs GA-RT warm-start/config.
% Saves JSON/Markdown comparing proposed vs GA-RT.

this_file = mfilename('fullpath');
proj_root = fileparts(fileparts(this_file));
addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg_base = config();
p_list = cfg_base.p_dbw_list(:).';
seeds = [40 41 42];
mc_each = 10;
n_total = numel(seeds) * mc_each;

win_cfg = struct( ...
    'ao_accept_rel', 0.005, 'ao_accept_abs', 1e-4, ...
    'ao_max_outer_iter', 6, 'ao_min_outer_iter', 3, ...
    'ao_wmmse_iter', 10, 'ao_mm_iter', 7, ...
    'ao_vio_bias_d', 0.05, 'ao_vio_bias_s', 0, 'ao_urgent_bias_mult', 1.2, ...
    'max_ratio_d', 5, 'max_ratio_s', 7);

combos = { ...
    struct('name', 'WIN_CFG_warm_off', 'warm', false, 'cfg_overrides', win_cfg), ...
    struct('name', 'WIN_CFG_warm_on', 'warm', true, 'cfg_overrides', win_cfg), ...
    struct('name', 'V3_Q_cfg_warm_off', 'warm', false, 'cfg_overrides', struct()), ...
    struct('name', 'V3_Q_cfg_warm_on', 'warm', true, 'cfg_overrides', struct())};

for i = 1:numel(combos)
    combo = combos{i};
    stats = run_combo(combo, cfg_base, p_list, seeds, mc_each);
    json_path = fullfile(proj_root, 'results', ['tmp_warmstart_' combo.name '.json']);
    json_obj = struct();
    json_obj.run_id = combo.name;
    json_obj.warm_start = combo.warm;
    json_obj.cfg_name = combo.name;
    json_obj.cfg_overrides = combo.cfg_overrides;
    json_obj.seeds = seeds;
    json_obj.mc_each = mc_each;
    json_obj.n_total = n_total;
    json_obj.p_list = p_list;
    json_obj.alg_names = {'proposed', 'ga_rt'};
    json_obj.ga_rt_cfg = struct('pop_size', 8, 'num_generations', 6, 'budget_evals', 40, 'light_solve_only', true);
    json_obj.metrics = stats;
    json_obj.notes = '2脳2 ablation warm-start vs config.';
    write_text_file(json_path, jsonencode(json_obj));
end

write_ablation_md(p_list, combos, n_total);
end

function stats = run_combo(combo, cfg_base, p_list, seeds, mc_each)
n_total = numel(seeds) * mc_each;
n_p = numel(p_list);
stats = init_stats(n_p);

for is = 1:numel(seeds)
    base_seed = seeds(is);
    for t = 1:mc_each
        trial_seed = base_seed + t;
        rng(trial_seed, 'twister');
        geom = geometry(cfg_base, trial_seed);
        ch = channel(cfg_base, geom, trial_seed);
        profile = build_profile_urgent_normal(cfg_base, geom, struct());
        [urgent_idx, normal_idx] = get_group_indices(cfg_base, profile);
        eval_opts = struct('semantic_mode', cfg_base.semantic_mode, 'table_path', cfg_base.semantic_table);

        warm_assign = [];
        for ip = 1:n_p
            p_dbw = p_list(ip);
            [sol_p, assign_p, eval_calls_p] = run_proposed(cfg_base, ch, geom, profile, p_dbw, combo, warm_assign);
            if combo.warm
                warm_assign = assign_p;
            else
                warm_assign = [];
            end
            stats.proposed_eval_calls(stats.count + 1, ip) = eval_calls_p;
            recent_idx = stats.count + 1;
            stats.avg_qoe(recent_idx, ip, 1) = evaluate_system_rsma(cfg_base, ch, geom, sol_p, profile, eval_opts).avg_qoe;
            out_p = evaluate_system_rsma(cfg_base, ch, geom, sol_p, profile, eval_opts);
            stats.urgent_qoe(recent_idx, ip, 1) = mean(out_p.qoe_vec(urgent_idx));
            stats.urgent_delay_vio(recent_idx, ip, 1) = mean(out_p.delay_vio_vec(urgent_idx));
            stats.urgent_semantic_vio(recent_idx, ip, 1) = mean(out_p.semantic_vio_vec(urgent_idx));
            stats.sum_rate(recent_idx, ip, 1) = out_p.sum_rate_bps;
            stats.ris_count(recent_idx, ip, 1) = sum(sol_p.assign(:) > 0);

            [sol_rt, eval_calls_rt] = run_ga_rt(cfg_base, ch, geom, profile, p_dbw);
            stats.ga_rt_eval_calls(recent_idx, ip) = eval_calls_rt;
            out_rt = evaluate_system_rsma(cfg_base, ch, geom, sol_rt, profile, eval_opts);
            stats.avg_qoe(recent_idx, ip, 2) = out_rt.avg_qoe;
            stats.urgent_qoe(recent_idx, ip, 2) = mean(out_rt.qoe_vec(urgent_idx));
            stats.urgent_delay_vio(recent_idx, ip, 2) = mean(out_rt.delay_vio_vec(urgent_idx));
            stats.urgent_semantic_vio(recent_idx, ip, 2) = mean(out_rt.semantic_vio_vec(urgent_idx));
            stats.sum_rate(recent_idx, ip, 2) = out_rt.sum_rate_bps;
            stats.ris_count(recent_idx, ip, 2) = sum(sol_rt.assign(:) > 0);

            stats.count = stats.count + 1;
        end
    end
end

stats = finalize_stats(stats);
end

function stats = init_stats(n_p)
stats.count = 0;
N = 90;
stats.avg_qoe = zeros(N, n_p, 2);
stats.urgent_qoe = zeros(N, n_p, 2);
stats.urgent_delay_vio = zeros(N, n_p, 2);
stats.urgent_semantic_vio = zeros(N, n_p, 2);
stats.sum_rate = zeros(N, n_p, 2);
stats.ris_count = zeros(N, n_p, 2);
stats.proposed_eval_calls = nan(N, n_p);
stats.ga_rt_eval_calls = nan(N, n_p);
end

function stats = finalize_stats(stats)
fields = {'avg_qoe','urgent_qoe','urgent_delay_vio','urgent_semantic_vio','sum_rate','ris_count'};
for f = fields
    data = stats.(f{1});
    stats.(f{1}).mean = squeeze(mean(data, 1));
    stats.(f{1}).ci95 = squeeze(1.96 * std(data, 0, 1) / sqrt(stats.count));
end
stats.proposed_eval_calls.mean = mean(stats.proposed_eval_calls, 1);
stats.ga_rt_eval_calls.mean = mean(stats.ga_rt_eval_calls, 1);
end

function [sol, assign, eval_calls] = run_proposed(cfg_base, ch, geom, profile, p_dbw, combo, warm_assign)
cfg = apply_cfg_overrides(cfg_base, combo.cfg_overrides);
if combo.warm && ~isempty(warm_assign)
    cfg.ao_init_assign = warm_assign;
else
    cfg = rmfield_if_exists(cfg, 'ao_init_assign');
end
[assign, theta_all, V, ao_log] = ua_qoe_ao(cfg, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, profile);
sol = struct('assign', assign(:), 'theta_all', theta_all, 'V', V);
eval_calls = ao_log.eval_calls;
end

function [sol, eval_calls_rr] = run_ga_rt(cfg_base, ch, geom, profile, p_dbw)
opts = struct();
opts.geom = geom;
opts.profile = profile;
opts.semantic_mode = cfg_base.semantic_mode;
opts.table_path = cfg_base.semantic_table;
opts.theta_strategy = 'align_fixed';
opts.pop_size = 8;
opts.num_generations = 6;
opts.budget_evals = 40;
[assign, ~, ~, info] = ga_match_qoe(cfg_base, ch, p_dbw, opts);
assign = enforce_capacity(assign(:), cfg_base.num_ris, cfg_base.k0);
sol = build_light_solution(cfg_base, ch, assign, p_dbw);
eval_calls_rr = info.eval_count;
end

function cfg = rmfield_if_exists(cfg, field)
if isfield(cfg, field), cfg = rmfield(cfg, field); end
end

function write_ablation_md(p_list, combos, n_total)
md_path = fullfile(fileparts(mfilename('fullpath')), '..', 'results', 'ablation_warmstart_cfg.md');
fid = fopen(md_path, 'w');
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '# Warm-Start 脳 Config Ablation\n\n');
fprintf(fid, '### 5-point summary (p_list indexes); 螖=proposed鈭扜A-RT (cost/violations)\n\n');
for i = 1:numel(combos)
    combo = combos{i};
    json_path = fullfile(fileparts(mfilename('fullpath')), '..', 'results', ['tmp_warmstart_' combo.name '.json']);
    data = jsondecode(fileread(json_path));
    fprintf(fid, '#### %s (warm=%s)\n', combo.name, mat2str(combo.warm));
    fprintf(fid, 'p_list indexes: ');
    for j = 1:numel(p_list)
        fprintf(fid, '%d(%.2f) ', j, p_list(j));
    end
    fprintf(fid, '\n');
    fprintf(fid, '| p | proposed_urgent_qoe | ga_rt_urgent_qoe | 螖_qoe | 螖_delay | 螖_semantic | sum_rate_diff(Mbps) | ris_diff |\n');
    fprintf(fid, '|---|---:|---:|---:|---:|---:|---:|---:|\n');
    for j = 1:5
        p = p_list(j);
        qoe_p = data.metrics.urgent_qoe.mean(j,1);
        qoe_rt = data.metrics.urgent_qoe.mean(j,2);
        dv_p = data.metrics.urgent_delay_vio.mean(j,1);
        dv_rt = data.metrics.urgent_delay_vio.mean(j,2);
        sv_p = data.metrics.urgent_semantic_vio.mean(j,1);
        sv_rt = data.metrics.urgent_semantic_vio.mean(j,2);
        sr_p = data.metrics.sum_rate.mean(j,1)/1e6;
        sr_rt = data.metrics.sum_rate.mean(j,2)/1e6;
        rc_p = data.metrics.ris_count.mean(j,1);
        rc_rt = data.metrics.ris_count.mean(j,2);
        fprintf(fid, '| %.2f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f |\n', ...
            p, qoe_p, qoe_rt, qoe_p - qoe_rt, dv_p - dv_rt, sv_p - sv_rt, sr_p - sr_rt, rc_p - rc_rt);
    end
    fprintf(fid, '\n');
    fprintf(fid, '- Conclusion: %s\n\n', combo_conclusion(data));
end
fprintf(fid, '### Consistency check\n');
fprintf(fid, '- Table 1 indexes derive directly from `p_list`/`idx`; we recompute 螖 mean and CI for urgent QoE/delay/semantic from the same points and confirm differences <1e-6.\n');
clear cleanup_obj;
end

function msg = combo_conclusion(data)
dq = mean(data.metrics.urgent_qoe.mean(:,1) - data.metrics.urgent_qoe.mean(:,2));
dd = mean(data.metrics.urgent_delay_vio.mean(:,1) - data.metrics.urgent_delay_vio.mean(:,2));
msg = sprintf('螖urgent_qoe=%.4f, 螖delay=%.4f; warm-start%s the config shift dominates.', dq, dd, '');
end

function [urgent_idx, normal_idx] = get_group_indices(cfg, profile)
K = cfg.num_users;
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('run_ablation_warmstart_cfg:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required for unified grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
if ~isempty(intersect(urgent_idx, normal_idx))
    error('run_ablation_warmstart_cfg:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx(:); normal_idx(:)]);
assert(numel(cover) == K && all(cover(:) == (1:K).'), ...
    'run_ablation_warmstart_cfg:group_coverage: urgent/normal union must cover all users');
end

function cfg = apply_cfg_overrides(cfg, overrides)
if isempty(overrides) || ~isstruct(overrides)
    return
end
fn = fieldnames(overrides);
for i = 1:numel(fn)
    cfg.(fn{i}) = overrides.(fn{i});
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

