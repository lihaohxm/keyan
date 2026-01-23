function plot_curves(varargin)
%PLOT_CURVES Plot curves from latest results.

p = inputParser;
addParameter(p, 'latest', true);
addParameter(p, 'run_id', '');
parse(p, varargin{:});

if ~exist('figures', 'dir')
    mkdir('figures');
end

if p.Results.latest
    files = dir(fullfile('results', '*_curves.mat'));
    if isempty(files)
        error('No results found.');
    end
    [~, idx] = max([files.datenum]);
    mat_path = fullfile(files(idx).folder, files(idx).name);
else
    mat_path = fullfile('results', [p.Results.run_id '_curves.mat']);
end

data = load(mat_path);
result = data.result;

if isfield(result, 'x_axis')
    x_vals = result.x_vals;
else
    x_vals = data.x_vals;
end

if isfield(result, 'sum_rate')
    figure('Visible', 'off');
    plot(x_vals, result.sum_rate, '-o');
    xlabel('x');
    ylabel('Sum-rate');
    legend(result.algorithms, 'Location', 'best');
    grid on;
    saveas(gcf, fullfile('figures', [data.run_id '_sum_rate.png']));
    close(gcf);

    figure('Visible', 'off');
    plot(x_vals, result.avg_qoe, '-o');
    xlabel('x');
    ylabel('Avg QoE');
    legend(result.algorithms, 'Location', 'best');
    grid on;
    saveas(gcf, fullfile('figures', [data.run_id '_avg_qoe.png']));
    close(gcf);
end

if isfield(result, 'weights')
    figure('Visible', 'off');
    plot(result.sum_rate, result.avg_qoe, '-o');
    xlabel('Sum-rate');
    ylabel('Avg QoE');
    legend(result.algorithms, 'Location', 'best');
    grid on;
    saveas(gcf, fullfile('figures', [data.run_id '_pareto.png']));
    close(gcf);
end
end
