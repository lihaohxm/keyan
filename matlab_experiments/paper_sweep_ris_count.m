function run_id = paper_sweep_ris_count(varargin)
%PAPER_SWEEP_RIS_COUNT Fair QoE-vs-RIS-count sweep.
% Usage:
%   paper_sweep_ris_count('mc',50,'seed',42,'p_dbw',-8,'ris_list',[1 2 3 4], ...
%       'out_name','ris_count_fixed')

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg = config();

p = inputParser;
addParameter(p, 'mc', 50);
addParameter(p, 'seed', 42);
addParameter(p, 'p_dbw', -8);
addParameter(p, 'ris_list', [1 2 3 4]);
addParameter(p, 'out_name', 'ris_count_fixed');
addParameter(p, 'ga_log', false);
addParameter(p, 'cfg_overrides', struct());
addParameter(p, 'save_figures', true);
addParameter(p, 'save_mat', true);
addParameter(p, 'save_csv', true);
addParameter(p, 'include_ga_ub', true);
addParameter(p, 'show_ga_ub', false);
addParameter(p, 'ga_rt_pop_size', 8);
addParameter(p, 'ga_rt_num_generations', 6);
addParameter(p, 'ga_rt_budget_evals', 40);
parse(p, varargin{:});

mc = p.Results.mc;
seed = p.Results.seed;
p_dbw = p.Results.p_dbw;
ris_list = p.Results.ris_list(:).';
out_name = char(p.Results.out_name);
cfg = apply_cfg_overrides(cfg, p.Results.cfg_overrides);

alg_names = {'proposed', 'random', 'norm', 'ga_rt'};
if p.Results.include_ga_ub
    alg_names{end + 1} = 'ga_ub';
end
num_alg = numel(alg_names);
num_ris_cfg = numel(ris_list);
tol_q = 1e-9;

urgent_qoe_all = zeros(mc, num_ris_cfg, num_alg);
avg_qoe_all = zeros(mc, num_ris_cfg, num_alg);
urgent_delay_vio_all = zeros(mc, num_ris_cfg, num_alg);
urgent_semantic_vio_all = zeros(mc, num_ris_cfg, num_alg);
sum_rate_all = zeros(mc, num_ris_cfg, num_alg);
urgent_sum_rate_all = zeros(mc, num_ris_cfg, num_alg);
urgent_avg_rate_all = zeros(mc, num_ris_cfg, num_alg);
ris_count_all = zeros(mc, num_ris_cfg, num_alg);
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

    for ix = 1:num_ris_cfg
        ris_count = ris_list(ix);

        cfg2 = cfg;
        cfg2.num_ris = ris_count;
        cfg2.ris_per_cell = ris_count;

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
                [assign, theta_all, V, ao_log] = ua_qoe_ao(cfg2, ch, geom, p_dbw, cfg2.semantic_mode, cfg2.semantic_table, profile);
                sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
                if isfield(ao_log, 'eval_calls')
                    proposed_budget = ao_log.eval_calls;
                else
                    proposed_budget = 100;
                end
                proposed_eval_calls_all(trial_idx, ix) = proposed_budget;
            else
                [assign_fixed, assign_info] = pick_assignment_local(cfg2, ch, geom, alg, profile, p_dbw, proposed_budget, p.Results.ga_log, p.Results);
                if strcmpi(alg, 'ga_rt')
                    % GA-RT uses real-time constrained GA and must stay in light-solve branch.
                    sol = build_light_solution(cfg2, ch, assign_fixed, p_dbw);
                    if isfield(assign_info, 'eval_count')
                        ga_rt_eval_calls_all(trial_idx, ix) = assign_info.eval_count;
                    end
                    if isfield(assign_info, 'budget_target')
                        ga_rt_budget_target_all(trial_idx, ix) = assign_info.budget_target;
                    end
                else
                    [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg2, ch, geom, p_dbw, assign_fixed, profile, struct());
                    sol = sol_fixed;
                end
            end

            out = evaluate_system_rsma(cfg2, ch, geom, sol, profile, opts_eval);
            rate_vec_bps = get_rate_vec_bps_compat(out, cfg2.num_users); % bps
            urgent_sum_rate_bps = sum(rate_vec_bps(urgent_idx)); % bps
            urgent_avg_rate_bps = mean(rate_vec_bps(urgent_idx)); % bps
            sanity = update_sanity_bounds(sanity, out, tol_q, sprintf('trial=%d,ris_idx=%d,alg=%s', trial_idx, ix, alg));
            urgent_qoe_all(trial_idx, ix, ia) = mean(out.qoe_vec(urgent_idx));
            avg_qoe_all(trial_idx, ix, ia) = out.avg_qoe;
            urgent_delay_vio_all(trial_idx, ix, ia) = mean(out.delay_vio_vec(urgent_idx));
            urgent_semantic_vio_all(trial_idx, ix, ia) = mean(out.semantic_vio_vec(urgent_idx));
            sum_rate_all(trial_idx, ix, ia) = out.sum_rate_bps;
            urgent_sum_rate_all(trial_idx, ix, ia) = urgent_sum_rate_bps;
            urgent_avg_rate_all(trial_idx, ix, ia) = urgent_avg_rate_bps;
            ris_count_all(trial_idx, ix, ia) = sum(sol.assign(:) > 0);
        end
    end

    if mod(trial_idx, 5) == 0 || trial_idx == mc
        fprintf('  trial %d/%d\n', trial_idx, mc);
    end
end

stats = struct();
[stats.urgent_qoe.mean, stats.urgent_qoe.ci95] = calc_mean_ci(urgent_qoe_all);
[stats.avg_qoe.mean, stats.avg_qoe.ci95] = calc_mean_ci(avg_qoe_all);
[stats.urgent_delay_vio.mean, stats.urgent_delay_vio.ci95] = calc_mean_ci(urgent_delay_vio_all);
[stats.urgent_semantic_vio.mean, stats.urgent_semantic_vio.ci95] = calc_mean_ci(urgent_semantic_vio_all);
[stats.sum_rate.mean, stats.sum_rate.ci95] = calc_mean_ci(sum_rate_all);
[stats.urgent_sum_rate.mean, stats.urgent_sum_rate.ci95] = calc_mean_ci(urgent_sum_rate_all);
[stats.urgent_avg_rate.mean, stats.urgent_avg_rate.ci95] = calc_mean_ci(urgent_avg_rate_all);
[stats.ris_count.mean, stats.ris_count.ci95] = calc_mean_ci(ris_count_all);
[stats.proposed_eval_calls.mean, stats.proposed_eval_calls.ci95] = calc_mean_ci2d(proposed_eval_calls_all);
[stats.ga_rt_eval_calls.mean, stats.ga_rt_eval_calls.ci95] = calc_mean_ci2d(ga_rt_eval_calls_all);
[stats.ga_rt_budget_target.mean, stats.ga_rt_budget_target.ci95] = calc_mean_ci2d(ga_rt_budget_target_all);
stats = ensure_urgent_rate_metrics(stats);

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

if p.Results.save_figures
    [plot_idx, plot_labels] = get_plot_view(alg_names, p.Results.show_ga_ub);
    plot_ci_lines(ris_list, stats.urgent_qoe.mean(:, plot_idx), stats.urgent_qoe.ci95(:, plot_idx), plot_labels, ...
        'Number of RIS (L)', 'Urgent QoE Cost', 'Urgent QoE Cost vs RIS Count', ...
        fullfile(fig_dir, [out_name '_urgent_qoe_cost.png']));

    plot_ci_lines(ris_list, stats.avg_qoe.mean(:, plot_idx), stats.avg_qoe.ci95(:, plot_idx), plot_labels, ...
        'Number of RIS (L)', 'Avg QoE Cost', 'Average QoE Cost vs RIS Count', ...
        fullfile(fig_dir, [out_name '_avg_qoe_cost.png']));
    plot_ci_lines(ris_list, stats.urgent_sum_rate.mean(:, plot_idx) / 1e6, stats.urgent_sum_rate.ci95(:, plot_idx) / 1e6, plot_labels, ...
        'Number of RIS (L)', 'Urgent Sum-Rate (Mbps)', 'Urgent Sum-Rate vs RIS Count', ...
        fullfile(fig_dir, [out_name '_urgent_sum_rate.png']));
end

run_id = out_name;
base = out_name;
mat_path = fullfile(res_dir, [base '.mat']);
json_path = fullfile(res_dir, [base '.json']);
csv_path = fullfile(res_dir, [base '.csv']);

if p.Results.save_mat
    save(mat_path, 'alg_names', 'ris_list', 'mc', 'seed', 'p_dbw', ...
        'urgent_qoe_all', 'avg_qoe_all', 'urgent_delay_vio_all', 'urgent_semantic_vio_all', ...
        'sum_rate_all', 'urgent_sum_rate_all', 'urgent_avg_rate_all', 'ris_count_all', ...
        'proposed_eval_calls_all', 'ga_rt_eval_calls_all', 'ga_rt_budget_target_all', ...
        'stats');
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
write_text_file(json_path, jsonencode(json_obj));

if p.Results.save_csv
    write_summary_csv(csv_path, ris_list, alg_names, stats);
end

fprintf('\nSaved:\n');
fprintf('  %s\n', fullfile(fig_dir, [out_name '_urgent_qoe_cost.png']));
fprintf('  %s\n', fullfile(fig_dir, [out_name '_avg_qoe_cost.png']));
fprintf('  %s\n', fullfile(fig_dir, [out_name '_urgent_sum_rate.png']));
fprintf('  %s\n', mat_path);
fprintf('  %s\n', json_path);
fprintf('  %s\n', csv_path);
end

function [mu, ci95] = calc_mean_ci(x)
mc = size(x, 1);
num_x = size(x, 2);
num_alg = size(x, 3);
mu = reshape(mean(x, 1), [num_x, num_alg]);
sd = reshape(std(x, 0, 1), [num_x, num_alg]);
ci95 = 1.96 * sd / sqrt(mc);
end

function plot_ci_lines(x, y_mean, y_ci, alg_names, x_label, y_label, fig_title, out_path)
A = numel(alg_names);
colors = lines(A);
markers = {'o', 's', 'd', '^', 'v', 'x'};
fig = figure('Color', 'w', 'Visible', 'off');
hold on;
for a = 1:A
    xa = x(:).';
    ya = y_mean(:, a).';
    plot(xa, ya, ['-' markers{a}], 'Color', colors(a, :), 'LineWidth', 1.6, ...
        'MarkerSize', 6, 'DisplayName', alg_names{a});
end
grid on;
xlabel(x_label);
ylabel(y_label);
title(fig_title);
xticks(x(:).');
legend('Location', 'best');
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
        opts.theta_strategy = 'random_fixed';
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
        opts.theta_strategy = 'random_fixed';
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
out = base_seed + trial_idx * 1000000 + ris_idx * 10000 + alg_idx * 100;
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
        lbl = 'GA-UB';
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
if any(q < -tol_q) || any(q > 1 + tol_q)
    error('paper_sweep_ris_count:qoe_out_of_range', 'QoE out of [0,1] at %s', ctx);
end
if any(qd < -tol_q) || any(qd > 1 + tol_q)
    error('paper_sweep_ris_count:qd_out_of_range', 'Qd out of [0,1] at %s', ctx);
end
if any(qs < -tol_q) || any(qs > 1 + tol_q)
    error('paper_sweep_ris_count:qs_out_of_range', 'Qs out of [0,1] at %s', ctx);
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
           'urgent_sum_rate', 'urgent_avg_rate', 'ris_count'};
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

function [mu, ci95] = calc_mean_ci2d(x)
mc = size(x, 1);
mu = mean(x, 1, 'omitnan');
sd = std(x, 0, 1, 'omitnan');
ci95 = 1.96 * sd / sqrt(mc);
mu = mu(:).';
ci95 = ci95(:).';
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
    stats.urgent_sum_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
if ~isfield(stats, 'urgent_avg_rate')
    stats.urgent_avg_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
end
