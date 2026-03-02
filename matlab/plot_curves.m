function plot_curves(run_id)
%PLOT_CURVES Plot sum-rate/QoE curves for a given run_id or all runs.
%   plot_curves(run_id) loads results/<run_id>_curves.mat and saves plots
%   to figures/. If run_id is empty or omitted, all runs are scanned.

    if nargin < 1
        run_id = "";
    end

    results_dir = "results";
    figures_dir = "figures";
    if ~exist(figures_dir, 'dir')
        mkdir(figures_dir);
    end

    if strlength(run_id) == 0
        files = dir(fullfile(results_dir, "*_curves.mat"));
        for k = 1:numel(files)
            [~, base_name] = fileparts(files(k).name);
            run_id = erase(base_name, "_curves");
            plot_single_run(results_dir, figures_dir, run_id);
        end
    else
        plot_single_run(results_dir, figures_dir, run_id);
    end
end

function plot_single_run(results_dir, figures_dir, run_id)
    mat_path = fullfile(results_dir, run_id + "_curves.mat");
    if ~exist(mat_path, 'file')
        warning("No curves file found for run_id: %s", run_id);
        return;
    end

    data = load(mat_path);
    if ~isfield(data, 'x_axis')
        warning("x_axis missing in %s", mat_path);
        return;
    end

    x_axis = data.x_axis;
    fields = fieldnames(data);

    sum_rate_fields = fields(startsWith(fields, "sum_rate_"));
    avg_qoe_fields = fields(startsWith(fields, "avg_qoe_"));

    if ~isempty(sum_rate_fields)
        fig = figure('Visible', 'off');
        hold on;
        for k = 1:numel(sum_rate_fields)
            values = data.(sum_rate_fields{k});
            plot(x_axis, values, '-o', 'LineWidth', 1.5, 'DisplayName', sum_rate_fields{k});
        end
        grid on;
        xlabel('x');
        ylabel('Sum Rate');
        title(run_id + " Sum Rate");
        legend('Location', 'best');
        saveas(fig, fullfile(figures_dir, run_id + "_sum_rate.png"));
        close(fig);
    end

    if ~isempty(avg_qoe_fields)
        fig = figure('Visible', 'off');
        hold on;
        for k = 1:numel(avg_qoe_fields)
            values = data.(avg_qoe_fields{k});
            plot(x_axis, values, '-s', 'LineWidth', 1.5, 'DisplayName', avg_qoe_fields{k});
        end
        grid on;
        xlabel('x');
        ylabel('Avg QoE');
        title(run_id + " Avg QoE");
        legend('Location', 'best');
        saveas(fig, fullfile(figures_dir, run_id + "_avg_qoe.png"));
        close(fig);
    end
end
