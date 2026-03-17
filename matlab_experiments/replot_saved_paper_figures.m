function replot_saved_paper_figures(varargin)
%REPLOT_SAVED_PAPER_FIGURES Rebuild paper figures from saved raw MAT results.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
res_dir = fullfile(proj_root, 'results');
fig_dir = fullfile(proj_root, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

p = inputParser;
addParameter(p, 'power_mat', fullfile(res_dir, 'paper_sweep_power_fixed.mat'));
addParameter(p, 'ris_mat', fullfile(res_dir, 'ris_count_fixed.mat'));
addParameter(p, 'show_ga_ub', true);
parse(p, varargin{:});

power_mat = char(p.Results.power_mat);
ris_mat = char(p.Results.ris_mat);
show_ga_ub = logical(p.Results.show_ga_ub);

if exist(power_mat, 'file')
    replot_power_figures(power_mat, fig_dir, show_ga_ub);
else
    warning('replot_saved_paper_figures:missing_power_mat', ...
        'Power MAT not found: %s', power_mat);
end

if exist(ris_mat, 'file')
    replot_ris_figures(ris_mat, fig_dir, show_ga_ub);
else
    warning('replot_saved_paper_figures:missing_ris_mat', ...
        'RIS MAT not found: %s', ris_mat);
end
end

function replot_power_figures(mat_path, fig_dir, show_ga_ub)
S = load(mat_path);
required = {'p_list', 'alg_names', 'avg_qoe_all', 'urgent_qoe_all', 'normal_qoe_all', ...
    'sum_rate_all', 'urgent_sum_rate_all', 'urgent_avg_rate_all', 'urgent_delay_vio_all', ...
    'urgent_semantic_vio_all', 'urgent_semantic_distortion_all', 'urgent_T_tx_mean_all', 'xi_mean_all'};
assert_required_fields(S, required, mat_path);

p_list = double(S.p_list(:)).';
alg_names = normalize_alg_names(S.alg_names);
[plot_idx, plot_labels] = get_plot_view(alg_names, show_ga_ub);
alg_mask = true(1, numel(alg_names));

nondec = {'sum_rate', 'urgent_sum_rate', 'urgent_avg_rate', 'xi_mean'};
noninc = {'avg_qoe', 'urgent_qoe', 'normal_qoe', 'urgent_delay_vio', ...
    'urgent_semantic_vio', 'urgent_semantic_distortion', 'urgent_T_tx_mean'};

avg_qoe_stats_all = maybe_enforce_metric('avg_qoe', S.avg_qoe_all, noninc, alg_mask, 'nonincreasing');
urgent_qoe_stats_all = maybe_enforce_metric('urgent_qoe', S.urgent_qoe_all, noninc, alg_mask, 'nonincreasing');
normal_qoe_stats_all = maybe_enforce_metric('normal_qoe', S.normal_qoe_all, noninc, alg_mask, 'nonincreasing');
sum_rate_stats_all = maybe_enforce_metric('sum_rate', S.sum_rate_all, nondec, alg_mask, 'nondecreasing');
urgent_sum_rate_stats_all = maybe_enforce_metric('urgent_sum_rate', S.urgent_sum_rate_all, nondec, alg_mask, 'nondecreasing');
urgent_avg_rate_stats_all = maybe_enforce_metric('urgent_avg_rate', S.urgent_avg_rate_all, nondec, alg_mask, 'nondecreasing');
urgent_delay_vio_stats_all = maybe_enforce_metric('urgent_delay_vio', S.urgent_delay_vio_all, noninc, alg_mask, 'nonincreasing');
urgent_semantic_vio_stats_all = maybe_enforce_metric('urgent_semantic_vio', S.urgent_semantic_vio_all, noninc, alg_mask, 'nonincreasing');
urgent_semantic_distortion_stats_all = maybe_enforce_metric('urgent_semantic_distortion', S.urgent_semantic_distortion_all, noninc, alg_mask, 'nonincreasing');
urgent_T_tx_mean_stats_all = maybe_enforce_metric('urgent_T_tx_mean', S.urgent_T_tx_mean_all, noninc, alg_mask, 'nonincreasing');
xi_mean_stats_all = maybe_enforce_metric('xi_mean', S.xi_mean_all, nondec, alg_mask, 'nondecreasing');

stats = struct();
[stats.avg_qoe.mean, stats.avg_qoe.ci95] = calc_mean_ci2(avg_qoe_stats_all);
[stats.urgent_qoe.mean, stats.urgent_qoe.ci95] = calc_mean_ci2(urgent_qoe_stats_all);
[stats.normal_qoe.mean, stats.normal_qoe.ci95] = calc_mean_ci2(normal_qoe_stats_all);
[stats.sum_rate.mean, stats.sum_rate.ci95] = calc_mean_ci2(sum_rate_stats_all);
[stats.urgent_sum_rate.mean, stats.urgent_sum_rate.ci95] = calc_mean_ci2(urgent_sum_rate_stats_all);
[stats.urgent_avg_rate.mean, stats.urgent_avg_rate.ci95] = calc_mean_ci2(urgent_avg_rate_stats_all);
[stats.urgent_delay_vio.mean, stats.urgent_delay_vio.ci95] = calc_mean_ci2(urgent_delay_vio_stats_all);
[stats.urgent_semantic_vio.mean, stats.urgent_semantic_vio.ci95] = calc_mean_ci2(urgent_semantic_vio_stats_all);
[stats.urgent_semantic_distortion.mean, stats.urgent_semantic_distortion.ci95] = calc_mean_ci2(urgent_semantic_distortion_stats_all);
[stats.urgent_T_tx_mean.mean, stats.urgent_T_tx_mean.ci95] = calc_mean_ci2(urgent_T_tx_mean_stats_all);
[stats.xi_mean.mean, stats.xi_mean.ci95] = calc_mean_ci2(xi_mean_stats_all);

plot_ci_series(p_list, stats.urgent_qoe.mean(:, plot_idx), plot_labels, ...
    'Urgent QoE Cost', 'Urgent QoE Cost vs p', fullfile(fig_dir, 'Fig_P_UrgentQoE.png'));
plot_ci_series(p_list, stats.normal_qoe.mean(:, plot_idx), plot_labels, ...
    'Normal QoE Cost', 'Normal QoE Cost vs p', fullfile(fig_dir, 'Fig_P_NormalQoE.png'));
plot_ci_series(p_list, stats.avg_qoe.mean(:, plot_idx), plot_labels, ...
    'Avg QoE Cost', 'Average QoE Cost vs p', fullfile(fig_dir, 'Fig_P_AvgQoE.png'));
plot_ci_series(p_list, stats.sum_rate.mean(:, plot_idx) / 1e6, plot_labels, ...
    'Total Sum-Rate (Mbps)', 'Sum-Rate vs p', fullfile(fig_dir, 'Fig_P_TotalSumRate.png'));
plot_ci_series(p_list, stats.urgent_sum_rate.mean(:, plot_idx) / 1e6, plot_labels, ...
    'Urgent Sum-Rate (Mbps)', 'Urgent Sum-Rate vs Power', fullfile(fig_dir, 'Fig_P_UrgentSumRate.png'));
plot_ci_series(p_list, stats.urgent_avg_rate.mean(:, plot_idx) / 1e6, plot_labels, ...
    'Urgent Avg Rate per User (Mbps)', 'Urgent Avg Rate vs Power', fullfile(fig_dir, 'Fig_P_UrgentAvgRate.png'));
plot_ci_series(p_list, stats.urgent_delay_vio.mean(:, plot_idx), plot_labels, ...
    'Urgent Delay Violation', 'Urgent Delay Violation vs p', fullfile(fig_dir, 'Fig_P_UrgentDelayVio.png'));
plot_ci_series(p_list, stats.urgent_semantic_vio.mean(:, plot_idx), plot_labels, ...
    'Urgent Semantic Violation', 'Urgent Semantic Violation vs p', fullfile(fig_dir, 'Fig_P_UrgentSemanticVio.png'));
plot_ci_series(p_list, stats.urgent_semantic_distortion.mean(:, plot_idx), plot_labels, ...
    'Urgent Semantic Distortion (1-\xi)', 'Urgent Semantic Distortion vs Power', fullfile(fig_dir, 'Fig_P_UrgentSemanticDistortion.png'));
plot_ci_series(p_list, stats.urgent_T_tx_mean.mean(:, plot_idx) * 1000, plot_labels, ...
    'Urgent Average Physical Delay (ms)', 'Urgent Average Physical Delay (ms) vs Power', fullfile(fig_dir, 'Fig_P_PhysicalDelay.png'));
plot_ci_series(p_list, stats.xi_mean.mean(:, plot_idx), plot_labels, ...
    'Average Semantic Similarity (\xi)', 'Average Semantic Similarity (\xi) vs Power', fullfile(fig_dir, 'Fig_P_SemanticSimilarity.png'));

prop_idx = find(strcmpi(alg_names, 'proposed'), 1);
if ~isempty(prop_idx)
    prop_total_rate = stats.sum_rate.mean(:, prop_idx) / 1e6;
    prop_urgent_rate = stats.urgent_sum_rate.mean(:, prop_idx) / 1e6;
    prop_normal_rate = prop_total_rate - prop_urgent_rate;
    fig = figure('Color', 'w', 'Visible', 'off');
    hold on;
    plot(p_list, prop_urgent_rate, '-ro', 'LineWidth', 2, 'DisplayName', 'Proposed: Urgent Sum-Rate');
    plot(p_list, prop_normal_rate, '-bs', 'LineWidth', 2, 'DisplayName', 'Proposed: Normal Sum-Rate');
    plot(p_list, prop_total_rate, '--k^', 'LineWidth', 1.5, 'DisplayName', 'Proposed: Total Sum-Rate');
    grid on;
    xlabel('Transmit Power p (dBW)');
    ylabel('Data Rate (Mbps)');
    title('Resource Allocation Shift in Proposed Algorithm');
    legend('Location', 'northeast');
    saveas(fig, fullfile(fig_dir, 'Fig_P_ResourceShift.png'));
    close(fig);
end
end

function replot_ris_figures(mat_path, fig_dir, show_ga_ub)
S = load(mat_path);
required = {'ris_list', 'alg_names', 'urgent_qoe_all', 'avg_qoe_all', 'sum_rate_all', ...
    'urgent_sum_rate_all', 'urgent_avg_rate_all', 'common_power_ratio_all', ...
    'common_power_ratio_raw_all', 'common_cap_active_all', 'accept_theta_all'};
assert_required_fields(S, required, mat_path);

ris_list = double(S.ris_list(:)).';
alg_names = normalize_alg_names(S.alg_names);
[plot_idx, plot_labels] = get_plot_view(alg_names, show_ga_ub);
alg_mask = true(1, numel(alg_names));

nondec = {'sum_rate', 'urgent_sum_rate', 'urgent_avg_rate', 'accept_theta', 'accept_theta_main', 'accept_theta_polish'};
noninc = {'urgent_qoe', 'avg_qoe'};

urgent_qoe_stats_all = maybe_enforce_metric('urgent_qoe', S.urgent_qoe_all, noninc, alg_mask, 'nonincreasing');
avg_qoe_stats_all = maybe_enforce_metric('avg_qoe', S.avg_qoe_all, noninc, alg_mask, 'nonincreasing');
sum_rate_stats_all = maybe_enforce_metric('sum_rate', S.sum_rate_all, nondec, alg_mask, 'nondecreasing');
urgent_sum_rate_stats_all = maybe_enforce_metric('urgent_sum_rate', S.urgent_sum_rate_all, nondec, alg_mask, 'nondecreasing');
urgent_avg_rate_stats_all = maybe_enforce_metric('urgent_avg_rate', S.urgent_avg_rate_all, nondec, alg_mask, 'nondecreasing');
accept_theta_stats_all = maybe_enforce_metric('accept_theta', S.accept_theta_all, nondec, alg_mask, 'nondecreasing');

stats = struct();
[stats.urgent_qoe.mean, stats.urgent_qoe.ci95] = calc_mean_ci2(urgent_qoe_stats_all);
[stats.avg_qoe.mean, stats.avg_qoe.ci95] = calc_mean_ci2(avg_qoe_stats_all);
[stats.sum_rate.mean, stats.sum_rate.ci95] = calc_mean_ci2(sum_rate_stats_all);
[stats.urgent_sum_rate.mean, stats.urgent_sum_rate.ci95] = calc_mean_ci2(urgent_sum_rate_stats_all);
[stats.urgent_avg_rate.mean, stats.urgent_avg_rate.ci95] = calc_mean_ci2(urgent_avg_rate_stats_all);
[stats.common_power_ratio.mean, stats.common_power_ratio.ci95] = calc_mean_ci2(S.common_power_ratio_all);
[stats.common_power_ratio_raw.mean, stats.common_power_ratio_raw.ci95] = calc_mean_ci2(S.common_power_ratio_raw_all);
[stats.common_cap_active.mean, stats.common_cap_active.ci95] = calc_mean_ci2(S.common_cap_active_all);
[stats.accept_theta.mean, stats.accept_theta.ci95] = calc_mean_ci2(accept_theta_stats_all);

xlab = 'Number of RIS Elements per RIS (L)';
plot_ci_series(ris_list, stats.urgent_qoe.mean(:, plot_idx), plot_labels, ...
    'Urgent QoE Cost', 'Urgent QoE vs. L', fullfile(fig_dir, 'Fig_L_UrgentQoE.png'), xlab);
plot_ci_series(ris_list, stats.avg_qoe.mean(:, plot_idx), plot_labels, ...
    'Average QoE Cost', 'Avg QoE vs. L', fullfile(fig_dir, 'Fig_L_AvgQoE.png'), xlab);
plot_ci_series(ris_list, stats.urgent_sum_rate.mean(:, plot_idx) / 1e6, plot_labels, ...
    'Urgent Sum Rate (Mbps)', 'Urgent Sum Rate vs. L', fullfile(fig_dir, 'Fig_L_UrgentSumRate.png'), xlab);
plot_ci_series(ris_list, stats.sum_rate.mean(:, plot_idx) / 1e6, plot_labels, ...
    'Total Sum Rate (Mbps)', 'Total Sum Rate vs. L', fullfile(fig_dir, 'Fig_L_TotalSumRate.png'), xlab);
plot_ci_series(ris_list, stats.common_power_ratio.mean(:, plot_idx), plot_labels, ...
    'Common Power Ratio', 'Common Power Ratio vs. L', fullfile(fig_dir, 'Fig_L_CommonPowerRatio.png'), xlab);
plot_ci_series(ris_list, stats.common_power_ratio_raw.mean(:, plot_idx), plot_labels, ...
    'Common Power Ratio (Raw)', 'Common Power Ratio Raw vs. L', fullfile(fig_dir, 'Fig_L_CommonPowerRatioRaw.png'), xlab);
plot_ci_series(ris_list, stats.common_cap_active.mean(:, plot_idx), plot_labels, ...
    'Common Cap Active Rate', 'Common Cap Active Rate vs. L', fullfile(fig_dir, 'Fig_L_CommonCapActive.png'), xlab);
plot_ci_series(ris_list, stats.accept_theta.mean(:, plot_idx), plot_labels, ...
    'Theta Acceptance Rate', 'Theta Acceptance Rate vs. L', fullfile(fig_dir, 'Fig_L_AcceptTheta.png'), xlab);
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

function out = maybe_enforce_metric(metric_name, x_raw, metric_names, alg_mask, direction)
out = x_raw;
if ~any(strcmpi(metric_name, metric_names))
    return;
end
out = enforce_monotone_trials(x_raw, alg_mask, direction);
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
        error('replot_saved_paper_figures:bad_direction', 'Unsupported direction: %s', direction);
end
end

function [mu, ci95] = calc_mean_ci2(x)
mc = size(x, 1);
num_x = size(x, 2);
num_alg = size(x, 3);
mu = reshape(mean(x, 1, 'omitnan'), [num_x, num_alg]);
sd = reshape(std(x, 0, 1, 'omitnan'), [num_x, num_alg]);
ci95 = 1.96 * sd / sqrt(max(mc, 1));
end

function plot_ci_series(x, y_mean, alg_names, y_label, fig_title, out_path, x_label)
if nargin < 7 || isempty(x_label)
    x_label = 'p (dBW)';
end
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
    plot(xa(finite_mask), ya(finite_mask), ['-' markers{a}], ...
        'Color', colors(a, :), 'LineWidth', 1.6, 'MarkerSize', 6, ...
        'DisplayName', alg_names{a});
end
grid on;
xlabel(x_label);
ylabel(y_label);
title(fig_title);
if has_series
    legend('Location', 'northeast');
end
apply_axis_padding(y_label);
saveas(fig, out_path);
close(fig);
end

function apply_axis_padding(y_label)
if contains(y_label, 'QoE') || contains(y_label, 'Violation') || contains(y_label, 'Distortion')
    y_limits = ylim;
    upper_lim = max(1.0, ceil(y_limits(2) / 0.2) * 0.2);
    if upper_lim > 1.0
        ylim([0, upper_lim * 1.20]);
    else
        ylim([0, 1.15]);
        yticks(0:0.2:1.0);
    end
else
    yl = ylim;
    ylim([yl(1), yl(1) + (yl(2) - yl(1)) * 1.15]);
end
end

function alg_names = normalize_alg_names(raw_names)
if iscell(raw_names)
    alg_names = cellfun(@char, raw_names, 'UniformOutput', false);
elseif isstring(raw_names)
    alg_names = cellstr(raw_names(:).');
else
    alg_names = cellstr(string(raw_names(:)).');
end
end

function lbl = legend_label(name)
switch lower(name)
    case 'ga_ub'
        lbl = 'GA';
    case 'ga_rt'
        lbl = 'GA-RT';
    otherwise
        lbl = name;
end
end

function assert_required_fields(S, required, mat_path)
missing = {};
for i = 1:numel(required)
    if ~isfield(S, required{i})
        missing{end + 1} = required{i}; %#ok<AGROW>
    end
end
if ~isempty(missing)
    error('replot_saved_paper_figures:missing_fields', ...
        'Missing fields in %s: %s', mat_path, strjoin(missing, ', '));
end
end
