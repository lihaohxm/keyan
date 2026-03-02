function plot_final_mean_lines()
%PLOT_FINAL_MEAN_LINES Plot final mean-only curves from final JSON files.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
res_dir = fullfile(proj_root, 'results');
fig_dir = fullfile(proj_root, 'figures');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end

jp = jsondecode(fileread(fullfile(res_dir, 'paper_sweep_power_ga_rt_final.json')));
jr = jsondecode(fileread(fullfile(res_dir, 'ris_count_ga_rt_final.json')));

alg_names = cellstr(string(jp.alg_names(:)).');
need = {'proposed','random','norm','ga_rt'};
keep = zeros(1, numel(need));
for i = 1:numel(need)
    keep(i) = find(strcmpi(alg_names, need{i}), 1);
end
plot_names = alg_names(keep);

p_list = double(jp.p_list(:)).';
mp = jp.metrics;
plot_mean_only(p_list, double(mp.urgent_qoe.mean(:, keep)), plot_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_final_urgent_qoe.png'), 'Power (dBW)', 'Urgent QoE Cost', 'Urgent QoE Cost vs Power');
plot_mean_only(p_list, double(mp.urgent_delay_vio.mean(:, keep)), plot_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_final_urgent_delay_vio.png'), 'Power (dBW)', 'Urgent Delay Violation', 'Urgent Delay Violation vs Power');
plot_mean_only(p_list, double(mp.urgent_semantic_vio.mean(:, keep)), plot_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_final_urgent_semantic_vio.png'), 'Power (dBW)', 'Urgent Semantic Violation', 'Urgent Semantic Violation vs Power');
plot_mean_only(p_list, double(mp.urgent_sum_rate.mean(:, keep)) / 1e6, plot_names, fullfile(fig_dir, 'paper_sweep_power_ga_rt_final_urgent_sum_rate.png'), 'Power (dBW)', 'Urgent Sum-Rate (Mbps)', 'Urgent Sum-Rate vs Power');

r_list = double(jr.ris_list(:)).';
mr = jr.metrics;
plot_mean_only(r_list, double(mr.urgent_qoe.mean(:, keep)), plot_names, fullfile(fig_dir, 'ris_count_ga_rt_final_urgent_qoe.png'), 'RIS Count', 'Urgent QoE Cost', 'Urgent QoE Cost vs RIS Count');
plot_mean_only(r_list, double(mr.urgent_delay_vio.mean(:, keep)), plot_names, fullfile(fig_dir, 'ris_count_ga_rt_final_urgent_delay_vio.png'), 'RIS Count', 'Urgent Delay Violation', 'Urgent Delay Violation vs RIS Count');
plot_mean_only(r_list, double(mr.urgent_semantic_vio.mean(:, keep)), plot_names, fullfile(fig_dir, 'ris_count_ga_rt_final_urgent_semantic_vio.png'), 'RIS Count', 'Urgent Semantic Violation', 'Urgent Semantic Violation vs RIS Count');
plot_mean_only(r_list, double(mr.urgent_sum_rate.mean(:, keep)) / 1e6, plot_names, fullfile(fig_dir, 'ris_count_ga_rt_final_urgent_sum_rate.png'), 'RIS Count', 'Urgent Sum-Rate (Mbps)', 'Urgent Sum-Rate vs RIS Count');
end

function plot_mean_only(x, y, alg_names, out_path, xlab, ylab, ttl)
fig = figure('Color', 'w', 'Visible', 'off');
hold on;
colors = lines(size(y, 2));
markers = {'o','s','d','^'};
for a = 1:size(y, 2)
    plot(x, y(:, a).', ['-' markers{a}], 'Color', colors(a, :), 'LineWidth', 1.6, 'MarkerSize', 5, ...
        'DisplayName', legend_name(alg_names{a}));
end
grid on;
xlabel(xlab);
ylabel(ylab);
title(ttl);
legend('Location', 'best');
saveas(fig, out_path);
close(fig);
end

function out = legend_name(in)
switch lower(in)
    case 'ga_rt'
        out = 'GA-RT';
    otherwise
        out = in;
end
end
