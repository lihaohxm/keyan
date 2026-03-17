function run_id = paper_ablation(varargin)
%PAPER_ABLATION Ablation study at fixed power.
% Usage:
%   paper_ablation('mc',50,'p_dbw',-20,'seed',42)
%
% Protocol:
% 1) rng(seed + trial_idx) once at each MC trial start.
% 2) No rng reset inside the trial.
% 3) Each trial generates geom/ch/profile once and reuses them for all groups.
%
% simple_beam setting in this script:
% - Enable cfg.ao_simple_beam=true (no RSMA-WMMSE update, fixed MRT beam).
% - Keep theta update enabled (cfg.ao_freeze_theta=false).
% - Keep random fallback enabled.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg = config();

p = inputParser;
addParameter(p, 'mc', 200);
addParameter(p, 'p_dbw', -20);
addParameter(p, 'seed', 42);
parse(p, varargin{:});

mc = p.Results.mc;
p_dbw = p.Results.p_dbw;
seed = p.Results.seed;

group_names = {'full', 'no_urgency', 'fixed_theta', 'simple_beam'};
num_g = numel(group_names);

urgent_qoe_all = zeros(mc, num_g);
avg_qoe_all = zeros(mc, num_g);
urgent_delay_vio_all = zeros(mc, num_g);
urgent_semantic_vio_all = zeros(mc, num_g);
sum_rate_all = zeros(mc, num_g);
urgent_sum_rate_all = zeros(mc, num_g);
tol_q = 1e-9;
sanity = struct( ...
    'qoe_min', inf, 'qoe_max', -inf, ...
    'Qd_min', inf, 'Qd_max', -inf, ...
    'Qs_min', inf, 'Qs_max', -inf);

fprintf('========================================\n');
fprintf('paper_ablation: MC=%d, p_dbw=%.2f, seed=%d\n', mc, p_dbw, seed);
fprintf('groups: %s\n', strjoin(group_names, ', '));
fprintf('========================================\n');

for trial_idx = 1:mc
    trial_seed = seed + trial_idx;
    rng(trial_seed, 'twister');

    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);
    profile = build_profile_urgent_normal(cfg, geom, struct());
    urgent_idx = get_urgent_indices(profile, cfg.num_users);

    for g = 1:num_g
        gname = group_names{g};
        [cfg2, profile2] = make_group_setup(cfg, profile, gname);
        [assign, theta_all, V, ~] = ua_qoe_ao(cfg2, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, profile2);
        sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
        out = evaluate_system_rsma(cfg2, ch, geom, sol, profile2, struct( ...
            'semantic_mode', cfg2.semantic_mode, 'table_path', cfg2.semantic_table));
        sanity = update_sanity_bounds(sanity, out, tol_q, sprintf('trial=%d,group=%s', trial_idx, gname));
        rate_vec_bps = get_rate_vec_bps_compat(out, cfg2.num_users);

        urgent_qoe_all(trial_idx, g) = mean(out.qoe_vec(urgent_idx));
        avg_qoe_all(trial_idx, g) = out.avg_qoe;
        urgent_delay_vio_all(trial_idx, g) = mean(out.delay_vio_vec(urgent_idx));
        urgent_semantic_vio_all(trial_idx, g) = mean(out.semantic_vio_vec(urgent_idx));
        sum_rate_all(trial_idx, g) = out.sum_rate_bps;
        urgent_sum_rate_all(trial_idx, g) = sum(rate_vec_bps(urgent_idx));
    end

    if mod(trial_idx, 10) == 0 || trial_idx == mc
        fprintf('  trial %d/%d\n', trial_idx, mc);
    end
end

function urgent_idx = get_urgent_indices(profile, K)
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx')
    error('paper_ablation:missing_groups', 'profile.groups.urgent_idx is required.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
if isempty(urgent_idx)
    error('paper_ablation:empty_urgent', 'profile.groups.urgent_idx resolved to empty.');
end
end

stats = struct();
[stats.urgent_qoe.mean, stats.urgent_qoe.ci95] = mean_ci(urgent_qoe_all);
[stats.avg_qoe.mean, stats.avg_qoe.ci95] = mean_ci(avg_qoe_all);
[stats.urgent_delay_vio.mean, stats.urgent_delay_vio.ci95] = mean_ci(urgent_delay_vio_all);
[stats.urgent_semantic_vio.mean, stats.urgent_semantic_vio.ci95] = mean_ci(urgent_semantic_vio_all);
[stats.sum_rate.mean, stats.sum_rate.ci95] = mean_ci(sum_rate_all);
[stats.urgent_sum_rate.mean, stats.urgent_sum_rate.ci95] = mean_ci(urgent_sum_rate_all);

fprintf('\nAblation Summary (mean +/- ci95)\n');
fprintf('%-12s | %-20s | %-20s | %-20s | %-20s\n', ...
    'group', 'urgent_qoe', 'urgent_delay_vio', 'urgent_sem_vio', 'sum_rate');
for g = 1:num_g
    fprintf('%-12s | %.6f +/- %.6f | %.6f +/- %.6f | %.6f +/- %.6f | %.6f +/- %.6f\n', ...
        group_names{g}, ...
        stats.urgent_qoe.mean(g), stats.urgent_qoe.ci95(g), ...
        stats.urgent_delay_vio.mean(g), stats.urgent_delay_vio.ci95(g), ...
        stats.urgent_semantic_vio.mean(g), stats.urgent_semantic_vio.ci95(g), ...
        stats.sum_rate.mean(g), stats.sum_rate.ci95(g));
end

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

plot_ablation(stats, group_names, fullfile(fig_dir, 'ablation.png'));

run_id = datestr(now, 'yyyymmdd_HHMMSS');
mat_path = fullfile(res_dir, ['ablation_' run_id '.mat']);
json_path = fullfile(res_dir, ['ablation_' run_id '.json']);

save(mat_path, ...
    'group_names', 'urgent_qoe_all', 'avg_qoe_all', 'urgent_delay_vio_all', ...
    'urgent_semantic_vio_all', 'sum_rate_all', 'urgent_sum_rate_all', ...
    'stats', 'mc', 'p_dbw', 'seed');

json_obj = struct();
json_obj.run_id = ['ablation_' run_id];
json_obj.mc = mc;
json_obj.p_dbw = p_dbw;
json_obj.seed = seed;
json_obj.groups = group_names;
json_obj.stats = stats;
json_obj.metrics = stats;
json_obj.sanity = sanity;
write_text_file(json_path, jsonencode(json_obj));

fprintf('\nSaved:\n');
fprintf('  %s\n', fullfile(fig_dir, 'ablation.png'));
fprintf('  %s\n', mat_path);
fprintf('  %s\n', json_path);
end

function [cfg2, profile2] = make_group_setup(cfg, profile, group_name)
cfg2 = cfg;
profile2 = profile;

switch lower(group_name)
    case 'full'
        cfg2.ao_freeze_theta = false;
        cfg2.ao_simple_beam = false;
        cfg2.ao_disable_random_fallback = false;
    case 'no_urgency'
        cfg2.ao_freeze_theta = false;
        cfg2.ao_simple_beam = false;
        cfg2.ao_disable_random_fallback = false;
        profile2.weights = repmat([0.5 0.5], cfg.num_users, 1);
        profile2.d_k(:) = cfg.deadlines(min(2, numel(cfg.deadlines)));
        profile2.dmax_k(:) = cfg.dmax;
    case 'fixed_theta'
        cfg2.ao_freeze_theta = true;
        cfg2.ao_disable_random_fallback = true;
        cfg2.ao_simple_beam = false;
    case 'simple_beam'
        cfg2.ao_simple_beam = true;
        cfg2.ao_freeze_theta = false;
        cfg2.ao_disable_random_fallback = false;
    otherwise
        error('Unknown group: %s', group_name);
end
end

function [mu, ci] = mean_ci(x)
mu = mean(x, 1);
sd = std(x, 0, 1);
ci = 1.96 * sd / sqrt(size(x, 1));
end

function plot_ablation(stats, group_names, out_path)
x = 1:numel(group_names);
fig = figure('Color', 'w', 'Visible', 'off');

subplot(2, 2, 1);
errorbar(x, stats.urgent_qoe.mean, stats.urgent_qoe.ci95, 'o-', 'LineWidth', 1.6);
grid on; xticks(x); xticklabels(group_names); xtickangle(15);
ylabel('Urgent QoE Cost'); title('Urgent QoE Cost');

subplot(2, 2, 2);
errorbar(x, stats.urgent_delay_vio.mean, stats.urgent_delay_vio.ci95, 's-', 'LineWidth', 1.6);
grid on; xticks(x); xticklabels(group_names); xtickangle(15);
ylabel('Urgent Delay Violation'); title('Urgent Delay Violation');

subplot(2, 2, 3);
errorbar(x, stats.urgent_semantic_vio.mean, stats.urgent_semantic_vio.ci95, 'd-', 'LineWidth', 1.6);
grid on; xticks(x); xticklabels(group_names); xtickangle(15);
ylabel('Urgent Semantic Violation'); title('Urgent Semantic Violation');

subplot(2, 2, 4);
errorbar(x, stats.sum_rate.mean, stats.sum_rate.ci95, '^-', 'LineWidth', 1.6);
grid on; xticks(x); xticklabels(group_names); xtickangle(15);
ylabel('Sum-Rate (bps)'); title('Sum-Rate');

sgtitle('Ablation at Fixed Power');
saveas(fig, out_path);
close(fig);
end

function sanity = update_sanity_bounds(sanity, out, tol_q, ctx)
q = out.qoe_vec(:);
qd = out.Qd_vec(:);
qs = out.Qs_vec(:);

if any(~isfinite(q)) || any(~isfinite(qd)) || any(~isfinite(qs))
    error('paper_ablation:nonfinite_qoe', 'Non-finite QoE/Qd/Qs at %s', ctx);
end
if any(qd < -tol_q)
    error('paper_ablation:qd_out_of_range', 'Qd out of [0,inf) at %s', ctx);
end
if any(qs < -tol_q)
    error('paper_ablation:qs_out_of_range', 'Qs out of [0,inf) at %s', ctx);
end

sanity.qoe_min = min(sanity.qoe_min, min(q));
sanity.qoe_max = max(sanity.qoe_max, max(q));
sanity.Qd_min = min(sanity.Qd_min, min(qd));
sanity.Qd_max = max(sanity.Qd_max, max(qd));
sanity.Qs_min = min(sanity.Qs_min, min(qs));
sanity.Qs_max = max(sanity.Qs_max, max(qs));
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
    error('paper_ablation:missing_rate_vec', 'Cannot recover per-user rate vector.');
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

function write_text_file(path_name, txt)
fid = fopen(path_name, 'w');
if fid < 0
    error('Cannot open file for writing: %s', path_name);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', txt);
clear cleanup_obj;
end
