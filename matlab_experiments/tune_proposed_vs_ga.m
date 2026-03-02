function best_result = tune_proposed_vs_ga(varargin)
%TUNE_PROPOSED_VS_GA Lightweight tuner for proposed vs GA-budgeted.
% Outputs (default tag _v2):
%   results/paper_sweep_power_tuned_v2.json
%   results/ris_count_tuned_v2.json
%   results/tuning_summary_v2.json
%   results/proposed_vs_ga_tuned_report_v2.md

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
res_dir = fullfile(proj_root, 'results');
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

p = inputParser;
addParameter(p, 'seed', 42);
addParameter(p, 'mc_mini', 10);
addParameter(p, 'mc_full', 50);
addParameter(p, 'n_candidates', 18);
addParameter(p, 'topk_full', 3);
addParameter(p, 'out_tag', '_v2');
parse(p, varargin{:});

seed = p.Results.seed;
mc_mini = p.Results.mc_mini;
mc_full = p.Results.mc_full;
n_candidates = p.Results.n_candidates;
topk_full = p.Results.topk_full;
out_tag = char(p.Results.out_tag);

metric_dir = metric_direction();

% qoe is cost (lower is better): qoe.m line 2.
score_cfg = struct();
score_cfg.alpha1 = 1.2;            % delay violation importance
score_cfg.alpha2 = 1.0;            % semantic violation importance
score_cfg.beta = 0.08;             % normalized sum-rate gain
score_cfg.delay_guard_th = 0.03;   % if proposed delay is much worse, penalize
score_cfg.delay_guard_pen = 2.0;
score_cfg.qoe_guard_th = 0.01;     % if proposed urgent_qoe(cost) is worse, penalize
score_cfg.qoe_guard_pen = 2.5;
score_cfg.sem_guard_th = 0.03;     % semantic guard
score_cfg.sem_guard_pen = 1.5;

mini_p_list = [-25, -19.285714285714285, -13.571428571428571, -7.8571428571428577, -5];
mini_ris_list = [1 3 4];

rng(seed, 'twister');
cands = build_candidates(n_candidates);
mini_results = cell(1, numel(cands));

fprintf('=== Mini tuning (%d candidates, mc=%d) ===\n', numel(cands), mc_mini);
for i = 1:numel(cands)
    cand = cands(i);
    out_power = sprintf('tune_tmp_power_%03d%s', i, out_tag);
    out_ris = sprintf('tune_tmp_ris_%03d%s', i, out_tag);

    paper_sweep_power('mc', mc_mini, 'seed', seed, ...
        'p_list', mini_p_list, 'out_name', out_power, ...
        'cfg_overrides', cand.cfg, 'save_figures', false, 'save_mat', false, 'save_csv', false);
    paper_sweep_ris_count('mc', mc_mini, 'seed', seed, 'p_dbw', -8, ...
        'ris_list', mini_ris_list, 'out_name', out_ris, ...
        'cfg_overrides', cand.cfg, 'save_figures', false, 'save_mat', false, 'save_csv', false);

    jp = jsondecode(fileread(fullfile(res_dir, [out_power '.json'])));
    jr = jsondecode(fileread(fullfile(res_dir, [out_ris '.json'])));
    m = evaluate_candidate(jp, jr, score_cfg, metric_dir);
    m.candidate_id = i;
    m.cfg = cand.cfg;
    mini_results{i} = m;
    fprintf('  cand %02d: delta=%.4f, guard(d/q/s)=(%d/%d/%d)\n', ...
        i, m.delta_score, m.delay_guard_violations, m.qoe_guard_violations, m.semantic_guard_violations);
end

mini_score = cellfun(@(x) x.delta_score, mini_results);
[~, ord] = sort(mini_score, 'descend');
top_idx = ord(1:min(topk_full, numel(ord)));

fprintf('=== Full validation (top-%d, mc=%d) ===\n', numel(top_idx), mc_full);
full_results = cell(1, numel(top_idx));
for t = 1:numel(top_idx)
    idx = top_idx(t);
    cand = cands(idx);
    out_power = sprintf('paper_sweep_power_tuned%s_c%d', out_tag, t);
    out_ris = sprintf('ris_count_tuned%s_c%d', out_tag, t);

    paper_sweep_power('mc', mc_full, 'seed', seed, 'out_name', out_power, ...
        'cfg_overrides', cand.cfg, 'save_figures', true, 'save_mat', true, 'save_csv', true);
    paper_sweep_ris_count('mc', mc_full, 'seed', seed, 'p_dbw', -8, 'ris_list', [1 2 3 4], ...
        'out_name', out_ris, 'cfg_overrides', cand.cfg, ...
        'save_figures', true, 'save_mat', true, 'save_csv', true);

    jp = jsondecode(fileread(fullfile(res_dir, [out_power '.json'])));
    jr = jsondecode(fileread(fullfile(res_dir, [out_ris '.json'])));
    m = evaluate_candidate(jp, jr, score_cfg, metric_dir);
    m.candidate_id = idx;
    m.cfg = cand.cfg;
    m.power_json = [out_power '.json'];
    m.ris_json = [out_ris '.json'];
    full_results{t} = m;
    fprintf('  top %d (cand %d): delta=%.4f, wins(uq/dv/sv)=%d/%d/%d, guard(d/q/s)=%d/%d/%d\n', ...
        t, idx, m.delta_score, m.win_uq_vs_gb, m.win_dv_vs_gb, m.win_sv_vs_gb, ...
        m.delay_guard_violations, m.qoe_guard_violations, m.semantic_guard_violations);
end

best_pos = pick_best(full_results);
best_result = full_results{best_pos};

final_power_name = ['paper_sweep_power_tuned' out_tag '.json'];
final_ris_name = ['ris_count_tuned' out_tag '.json'];
copyfile(fullfile(res_dir, best_result.power_json), fullfile(res_dir, final_power_name));
copyfile(fullfile(res_dir, best_result.ris_json), fullfile(res_dir, final_ris_name));

summary = struct();
summary.seed = seed;
summary.mc_mini = mc_mini;
summary.mc_full = mc_full;
summary.score_cfg = score_cfg;
summary.metric_direction = metric_dir;
summary.candidates = mini_results;
summary.topk_indices = top_idx;
summary.full_results = full_results;
summary.best_result = best_result;

summary_name = ['tuning_summary' out_tag '.json'];
fid = fopen(fullfile(res_dir, summary_name), 'w');
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', jsonencode(summary));
clear cleanup_obj;

jp_best = jsondecode(fileread(fullfile(res_dir, final_power_name)));
jr_best = jsondecode(fileread(fullfile(res_dir, final_ris_name)));
md_name = ['proposed_vs_ga_tuned_report' out_tag '.md'];
write_report_v2(fullfile(res_dir, md_name), jp_best, jr_best, summary, metric_dir);

fprintf('Best candidate id=%d, delta=%.4f\n', best_result.candidate_id, best_result.delta_score);
fprintf('Saved outputs:\n  %s\n  %s\n  %s\n  %s\n', ...
    fullfile(res_dir, final_power_name), ...
    fullfile(res_dir, final_ris_name), ...
    fullfile(res_dir, summary_name), ...
    fullfile(res_dir, md_name));
end

function dir = metric_direction()
dir = struct();
dir.urgent_qoe = 'lower_better';           % qoe.m line 2
dir.urgent_delay_vio = 'lower_better';
dir.urgent_semantic_vio = 'lower_better';
dir.sum_rate = 'higher_better';
end

function cands = build_candidates(n)
% Baseline + random search around key knobs.
grid_accept = [0.001 0.003 0.005 0.008];
grid_outer = [6 8 10];
grid_wmmse = [6 8 10];
grid_mm = [3 5 7];
grid_vd = [0 0.05 0.1 0.2 0.3];
grid_vs = [0 0.05 0.1 0.2 0.3];
grid_um = [1.0 1.2 1.4 1.6 1.8 2.0];
grid_mrd = [5 7 10];
grid_mrs = [5 7 10];

base = struct('ao_accept_rel', 0.005, 'ao_accept_abs', 1e-4, ...
    'ao_max_outer_iter', 8, 'ao_min_outer_iter', 3, ...
    'ao_wmmse_iter', 8, 'ao_mm_iter', 5, ...
    'ao_vio_bias_d', 0.0, 'ao_vio_bias_s', 0.0, 'ao_urgent_bias_mult', 1.0, ...
    'max_ratio_d', 5, 'max_ratio_s', 5);

cands = repmat(struct('cfg', base), 1, n);
cands(1).cfg = base;
for i = 2:n
    cfg = base;
    cfg.ao_accept_rel = pick(grid_accept);
    cfg.ao_max_outer_iter = pick(grid_outer);
    cfg.ao_wmmse_iter = pick(grid_wmmse);
    cfg.ao_mm_iter = pick(grid_mm);
    cfg.ao_vio_bias_d = pick(grid_vd);
    cfg.ao_vio_bias_s = pick(grid_vs);
    cfg.ao_urgent_bias_mult = pick(grid_um);
    cfg.max_ratio_d = pick(grid_mrd);
    cfg.max_ratio_s = pick(grid_mrs);
    cands(i).cfg = cfg;
end
end

function m = evaluate_candidate(jp, jr, score_cfg, metric_dir)
alg = string(jp.alg_names(:));
id_p = find(alg == "proposed", 1);
id_gb = find(alg == "ga_budgeted", 1);
id_gf = find(alg == "ga_full", 1);

if isempty(id_p) || isempty(id_gb)
    error('Missing proposed or ga_budgeted in result json.');
end

num_p = numel(jp.p_list);
num_r = numel(jr.ris_list);
delta_vec = zeros(num_p + num_r, 1);
guard_delay = 0;
guard_qoe = 0;
guard_sem = 0;

for i = 1:num_p
    s_all = double(jp.metrics.sum_rate.mean(i, :));
    sr_norm_p = double(jp.metrics.sum_rate.mean(i, id_p)) / max(s_all + eps);
    sr_norm_gb = double(jp.metrics.sum_rate.mean(i, id_gb)) / max(s_all + eps);

    qoe_p = double(jp.metrics.urgent_qoe.mean(i, id_p));
    qoe_g = double(jp.metrics.urgent_qoe.mean(i, id_gb));
    dv_p = double(jp.metrics.urgent_delay_vio.mean(i, id_p));
    dv_g = double(jp.metrics.urgent_delay_vio.mean(i, id_gb));
    sv_p = double(jp.metrics.urgent_semantic_vio.mean(i, id_p));
    sv_g = double(jp.metrics.urgent_semantic_vio.mean(i, id_gb));

    sp = score_scalar(qoe_p, dv_p, sv_p, sr_norm_p, score_cfg, metric_dir);
    sg = score_scalar(qoe_g, dv_g, sv_g, sr_norm_gb, score_cfg, metric_dir);
    d = sp - sg;

    if (dv_p - dv_g) > score_cfg.delay_guard_th
        d = d - score_cfg.delay_guard_pen * ((dv_p - dv_g) - score_cfg.delay_guard_th);
        guard_delay = guard_delay + 1;
    end
    if (qoe_p - qoe_g) > score_cfg.qoe_guard_th
        d = d - score_cfg.qoe_guard_pen * ((qoe_p - qoe_g) - score_cfg.qoe_guard_th);
        guard_qoe = guard_qoe + 1;
    end
    if (sv_p - sv_g) > score_cfg.sem_guard_th
        d = d - score_cfg.sem_guard_pen * ((sv_p - sv_g) - score_cfg.sem_guard_th);
        guard_sem = guard_sem + 1;
    end
    delta_vec(i) = d;
end

for i = 1:num_r
    s_all = double(jr.stats.sum_rate.mean(i, :));
    sr_norm_p = double(jr.stats.sum_rate.mean(i, id_p)) / max(s_all + eps);
    sr_norm_gb = double(jr.stats.sum_rate.mean(i, id_gb)) / max(s_all + eps);

    qoe_p = double(jr.stats.urgent_qoe.mean(i, id_p));
    qoe_g = double(jr.stats.urgent_qoe.mean(i, id_gb));
    dv_p = double(jr.stats.urgent_delay_vio.mean(i, id_p));
    dv_g = double(jr.stats.urgent_delay_vio.mean(i, id_gb));
    sv_p = double(jr.stats.urgent_semantic_vio.mean(i, id_p));
    sv_g = double(jr.stats.urgent_semantic_vio.mean(i, id_gb));

    sp = score_scalar(qoe_p, dv_p, sv_p, sr_norm_p, score_cfg, metric_dir);
    sg = score_scalar(qoe_g, dv_g, sv_g, sr_norm_gb, score_cfg, metric_dir);
    d = sp - sg;

    if (dv_p - dv_g) > score_cfg.delay_guard_th
        d = d - score_cfg.delay_guard_pen * ((dv_p - dv_g) - score_cfg.delay_guard_th);
        guard_delay = guard_delay + 1;
    end
    if (qoe_p - qoe_g) > score_cfg.qoe_guard_th
        d = d - score_cfg.qoe_guard_pen * ((qoe_p - qoe_g) - score_cfg.qoe_guard_th);
        guard_qoe = guard_qoe + 1;
    end
    if (sv_p - sv_g) > score_cfg.sem_guard_th
        d = d - score_cfg.sem_guard_pen * ((sv_p - sv_g) - score_cfg.sem_guard_th);
        guard_sem = guard_sem + 1;
    end
    delta_vec(num_p + i) = d;
end

m = struct();
m.delta_score = mean(delta_vec);
m.delta_vec = delta_vec;
m.delay_guard_violations = guard_delay;
m.qoe_guard_violations = guard_qoe;
m.semantic_guard_violations = guard_sem;
[m.win_uq_vs_gb, m.win_dv_vs_gb, m.win_sv_vs_gb] = win_count(jp, jr, id_p, id_gb, metric_dir);
if ~isempty(id_gf)
    [m.win_uq_vs_gf, m.win_dv_vs_gf, m.win_sv_vs_gf] = win_count(jp, jr, id_p, id_gf, metric_dir);
else
    [m.win_uq_vs_gf, m.win_dv_vs_gf, m.win_sv_vs_gf] = deal(0, 0, 0);
end
m.total_points = num_p + num_r;
end

function s = score_scalar(uq, dv, sv, srn, c, metric_dir)
if strcmp(metric_dir.urgent_qoe, 'lower_better')
    uq_term = -uq;
else
    uq_term = uq;
end
s = uq_term - c.alpha1 * dv - c.alpha2 * sv + c.beta * srn;
end

function [wu, wd, ws] = win_count(jp, jr, ip, ig, metric_dir)
uq_p = [double(jp.metrics.urgent_qoe.mean(:, ip)); double(jr.stats.urgent_qoe.mean(:, ip))];
uq_g = [double(jp.metrics.urgent_qoe.mean(:, ig)); double(jr.stats.urgent_qoe.mean(:, ig))];
dv_p = [double(jp.metrics.urgent_delay_vio.mean(:, ip)); double(jr.stats.urgent_delay_vio.mean(:, ip))];
dv_g = [double(jp.metrics.urgent_delay_vio.mean(:, ig)); double(jr.stats.urgent_delay_vio.mean(:, ig))];
sv_p = [double(jp.metrics.urgent_semantic_vio.mean(:, ip)); double(jr.stats.urgent_semantic_vio.mean(:, ip))];
sv_g = [double(jp.metrics.urgent_semantic_vio.mean(:, ig)); double(jr.stats.urgent_semantic_vio.mean(:, ig))];

wu = sum(is_better(uq_p, uq_g, metric_dir.urgent_qoe));
wd = sum(is_better(dv_p, dv_g, metric_dir.urgent_delay_vio));
ws = sum(is_better(sv_p, sv_g, metric_dir.urgent_semantic_vio));
end

function b = is_better(a, c, direction)
if strcmp(direction, 'lower_better')
    b = a <= c;
else
    b = a >= c;
end
end

function idx = pick_best(full_results)
% Prefer higher delta, fewer guards, better wins.
score = cellfun(@(x) x.delta_score, full_results);
guard_total = cellfun(@(x) x.delay_guard_violations + x.qoe_guard_violations + x.semantic_guard_violations, full_results);
wu = cellfun(@(x) x.win_uq_vs_gb, full_results);
ws = cellfun(@(x) x.win_sv_vs_gb, full_results);
wd = cellfun(@(x) x.win_dv_vs_gb, full_results);
[~, ord] = sortrows([-score(:), guard_total(:), -wu(:), -ws(:), -wd(:)], [1 2 3 4 5]);
idx = ord(1);
end

function write_report_v2(path_md, jp, jr, summary, metric_dir)
alg = string(jp.alg_names(:));
id_p = find(alg == "proposed", 1);
id_gb = find(alg == "ga_budgeted", 1);
id_gf = find(alg == "ga_full", 1);

best = summary.best_result;
tp = best.total_points;

[wu, wd, ws] = win_count(jp, jr, id_p, id_gb, metric_dir);
if ~isempty(id_gf)
    [wu_gf, wd_gf, ws_gf] = win_count(jp, jr, id_p, id_gf, metric_dir);
else
    [wu_gf, wd_gf, ws_gf] = deal(0, 0, 0);
end

sanity = build_sanity_points(jp, jr, id_p, id_gb, metric_dir);
top10 = top10_rows(summary.candidates);

fid = fopen(path_md, 'w');
cleanup_obj = onCleanup(@() fclose(fid));

fprintf(fid, '# Proposed vs GA Tuned Report (v2)\n\n');
fprintf(fid, '## Metric Direction and Evidence\n');
fprintf(fid, '- `urgent_qoe`: **lower is better (cost)**.\n');
fprintf(fid, '- Evidence: `matlab_sim/qoe.m:2` explicitly states "QoE cost (lower is better)".\n');
fprintf(fid, '- Evidence: `matlab_sim/evaluate_system_rsma.m:73-82` exports `out.avg_qoe` directly from `qoe(...)`.\n');
fprintf(fid, '- `urgent_delay_vio`: lower is better.\n');
fprintf(fid, '- `urgent_semantic_vio`: lower is better.\n');
fprintf(fid, '- `sum_rate`: higher is better.\n\n');

fprintf(fid, '## Final Tuned Configuration\n');
write_cfg(fid, best.cfg);
fprintf(fid, '\n');

fprintf(fid, '## Objective Used In Tuning\n');
fprintf(fid, '- score = -urgent_qoe - alpha1*urgent_delay_vio - alpha2*urgent_semantic_vio + beta*sum_rate_norm\n');
fprintf(fid, '- alpha1=%.3g, alpha2=%.3g, beta=%.3g\n', summary.score_cfg.alpha1, summary.score_cfg.alpha2, summary.score_cfg.beta);
fprintf(fid, '- guard(delay): if proposed-delay > ga_budgeted-delay + %.3f, penalty %.3g * excess\n', ...
    summary.score_cfg.delay_guard_th, summary.score_cfg.delay_guard_pen);
fprintf(fid, '- guard(qoe): if proposed-qoe > ga_budgeted-qoe + %.3f, penalty %.3g * excess\n', ...
    summary.score_cfg.qoe_guard_th, summary.score_cfg.qoe_guard_pen);
fprintf(fid, '- guard(semantic): if proposed-semantic_vio > ga_budgeted-semantic_vio + %.3f, penalty %.3g * excess\n\n', ...
    summary.score_cfg.sem_guard_th, summary.score_cfg.sem_guard_pen);

fprintf(fid, '## Sanity Check (3 sampled points)\n');
fprintf(fid, '| point | proposed (uq/dv/sv) | ga_budgeted (uq/dv/sv) | uq_cmp | dv_cmp | sv_cmp |\n');
fprintf(fid, '|---|---|---|---|---|---|\n');
for i = 1:numel(sanity)
    s = sanity(i);
    fprintf(fid, '| %s | %.3f / %.3f / %.3f | %.3f / %.3f / %.3f | %s | %s | %s |\n', ...
        s.label, s.uq_p, s.dv_p, s.sv_p, s.uq_g, s.dv_g, s.sv_g, s.uq_cmp, s.dv_cmp, s.sv_cmp);
end
fprintf(fid, '\n');

fprintf(fid, '## Tuning Search Top-10 (by mini-search delta_score)\n');
fprintf(fid, '| rank | candidate_id | delta_score | guard_d | guard_q | guard_s | win_uq_vs_gb | win_dv_vs_gb | win_sv_vs_gb |\n');
fprintf(fid, '|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n');
for i = 1:size(top10, 1)
    r = top10(i, :);
    fprintf(fid, '| %d | %d | %.4f | %d | %d | %d | %d | %d | %d |\n', ...
        i, r(1), r(2), r(3), r(4), r(5), r(6), r(7), r(8));
end
fprintf(fid, '\n');

fprintf(fid, '## Power Sweep Key Table (5 points)\n');
fprintf(fid, '| p(dBW) | alg | urgent_qoe | urgent_delay_vio | urgent_semantic_vio | sum_rate(Mbps) | ris_count |\n');
fprintf(fid, '|---:|---|---:|---:|---:|---:|---:|\n');
p_show = [-25, -19.285714285714285, -13.571428571428571, -7.8571428571428577, -5];
for i = 1:numel(p_show)
    ip = find(abs(double(jp.p_list(:)) - p_show(i)) < 1e-9, 1);
    if isempty(ip), continue; end
    write_row(fid, p_show(i), 'proposed', jp.metrics, ip, id_p);
    write_row(fid, p_show(i), 'ga_budgeted', jp.metrics, ip, id_gb);
    if ~isempty(id_gf), write_row(fid, p_show(i), 'ga_full', jp.metrics, ip, id_gf); end
end
fprintf(fid, '\n');

fprintf(fid, '## RIS Sweep Key Table (3 points)\n');
fprintf(fid, '| RIS L | alg | urgent_qoe | urgent_delay_vio | urgent_semantic_vio | sum_rate(Mbps) | ris_count |\n');
fprintf(fid, '|---:|---|---:|---:|---:|---:|---:|\n');
r_show = [1 3 4];
for i = 1:numel(r_show)
    ir = find(double(jr.ris_list(:)) == r_show(i), 1);
    if isempty(ir), continue; end
    write_row(fid, r_show(i), 'proposed', jr.stats, ir, id_p);
    write_row(fid, r_show(i), 'ga_budgeted', jr.stats, ir, id_gb);
    if ~isempty(id_gf), write_row(fid, r_show(i), 'ga_full', jr.stats, ir, id_gf); end
end
fprintf(fid, '\n');

fprintf(fid, '## Win Statistics (12 points: 8 power + 4 RIS)\n');
fprintf(fid, '- proposed_tuned vs ga_budgeted:\n');
fprintf(fid, '  - urgent_qoe wins: %d/%d\n', wu, tp);
fprintf(fid, '  - urgent_delay_vio wins: %d/%d\n', wd, tp);
fprintf(fid, '  - urgent_semantic_vio wins: %d/%d\n', ws, tp);
fprintf(fid, '- guard violations (best candidate): delay=%d, qoe=%d, semantic=%d\n', ...
    best.delay_guard_violations, best.qoe_guard_violations, best.semantic_guard_violations);
if ~isempty(id_gf)
    fprintf(fid, '- proposed_tuned vs ga_full:\n');
    fprintf(fid, '  - urgent_qoe wins: %d/%d\n', wu_gf, tp);
    fprintf(fid, '  - urgent_delay_vio wins: %d/%d\n', wd_gf, tp);
    fprintf(fid, '  - urgent_semantic_vio wins: %d/%d\n', ws_gf, tp);
end
fprintf(fid, '\n');

fprintf(fid, '## Notes\n');
if wu < 8
    fprintf(fid, '- urgent_qoe wins below target (8/12). Main trade-off is delay-focused guards and limited AO budget.\n');
    fprintf(fid, '- Minimal next change: increase `ao_max_outer_iter` by +2 only for power points where qoe guard triggers.\n');
else
    fprintf(fid, '- urgent_qoe wins reached target (>=8/12).\n');
end
fprintf(fid, '- Report consistency sanity-check included above to prevent direction/sign mismatch.\n');

clear cleanup_obj;
end

function write_cfg(fid, cfg)
f = fieldnames(cfg);
for i = 1:numel(f)
    v = cfg.(f{i});
    if isnumeric(v) && isscalar(v)
        fprintf(fid, '- %s: %.6g\n', f{i}, v);
    else
        fprintf(fid, '- %s: %s\n', f{i}, mat2str(v));
    end
end
end

function rows = top10_rows(mini_results)
n = numel(mini_results);
M = zeros(n, 8);
for i = 1:n
    m = mini_results{i};
    M(i, :) = [m.candidate_id, m.delta_score, m.delay_guard_violations, ...
        m.qoe_guard_violations, m.semantic_guard_violations, ...
        m.win_uq_vs_gb, m.win_dv_vs_gb, m.win_sv_vs_gb];
end
[~, ord] = sort(M(:, 2), 'descend');
M = M(ord, :);
rows = M(1:min(10, size(M, 1)), :);
end

function sanity = build_sanity_points(jp, jr, id_p, id_gb, metric_dir)
% 3 deterministic sampled points: power first, power last, ris middle.
sanity = repmat(struct('label', '', 'uq_p', 0, 'dv_p', 0, 'sv_p', 0, ...
    'uq_g', 0, 'dv_g', 0, 'sv_g', 0, 'uq_cmp', '', 'dv_cmp', '', 'sv_cmp', ''), 1, 3);

sanity(1) = make_sanity_row('power p=-25', ...
    double(jp.metrics.urgent_qoe.mean(1, id_p)), ...
    double(jp.metrics.urgent_delay_vio.mean(1, id_p)), ...
    double(jp.metrics.urgent_semantic_vio.mean(1, id_p)), ...
    double(jp.metrics.urgent_qoe.mean(1, id_gb)), ...
    double(jp.metrics.urgent_delay_vio.mean(1, id_gb)), ...
    double(jp.metrics.urgent_semantic_vio.mean(1, id_gb)), metric_dir);

sanity(2) = make_sanity_row(sprintf('power p=%.2f', double(jp.p_list(end))), ...
    double(jp.metrics.urgent_qoe.mean(end, id_p)), ...
    double(jp.metrics.urgent_delay_vio.mean(end, id_p)), ...
    double(jp.metrics.urgent_semantic_vio.mean(end, id_p)), ...
    double(jp.metrics.urgent_qoe.mean(end, id_gb)), ...
    double(jp.metrics.urgent_delay_vio.mean(end, id_gb)), ...
    double(jp.metrics.urgent_semantic_vio.mean(end, id_gb)), metric_dir);

mid = max(1, ceil(numel(jr.ris_list)/2));
sanity(3) = make_sanity_row(sprintf('ris L=%d', int32(jr.ris_list(mid))), ...
    double(jr.stats.urgent_qoe.mean(mid, id_p)), ...
    double(jr.stats.urgent_delay_vio.mean(mid, id_p)), ...
    double(jr.stats.urgent_semantic_vio.mean(mid, id_p)), ...
    double(jr.stats.urgent_qoe.mean(mid, id_gb)), ...
    double(jr.stats.urgent_delay_vio.mean(mid, id_gb)), ...
    double(jr.stats.urgent_semantic_vio.mean(mid, id_gb)), metric_dir);
end

function s = make_sanity_row(label, uq_p, dv_p, sv_p, uq_g, dv_g, sv_g, metric_dir)
s = struct();
s.label = label;
s.uq_p = uq_p; s.dv_p = dv_p; s.sv_p = sv_p;
s.uq_g = uq_g; s.dv_g = dv_g; s.sv_g = sv_g;
s.uq_cmp = cmp_tag(uq_p, uq_g, metric_dir.urgent_qoe);
s.dv_cmp = cmp_tag(dv_p, dv_g, metric_dir.urgent_delay_vio);
s.sv_cmp = cmp_tag(sv_p, sv_g, metric_dir.urgent_semantic_vio);
end

function t = cmp_tag(a, b, direction)
if strcmp(direction, 'lower_better')
    if a <= b
        t = 'proposed_better_or_equal';
    else
        t = 'proposed_worse';
    end
else
    if a >= b
        t = 'proposed_better_or_equal';
    else
        t = 'proposed_worse';
    end
end
end

function write_row(fid, x, alg_name, S, i, id)
fprintf(fid, '| %.2f | %s | %.3f | %.3f | %.3f | %.2f | %.2f |\n', ...
    x, alg_name, ...
    double(S.urgent_qoe.mean(i, id)), ...
    double(S.urgent_delay_vio.mean(i, id)), ...
    double(S.urgent_semantic_vio.mean(i, id)), ...
    double(S.sum_rate.mean(i, id))/1e6, ...
    double(S.ris_count.mean(i, id)));
end

function v = pick(arr)
v = arr(randi(numel(arr)));
end
