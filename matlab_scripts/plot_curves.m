function plot_curves(varargin)
%PLOT_CURVES Clean publication-ready plots. PNG only, no error bars.
%
% Usage:
%   plot_curves('latest', true)
%   plot_curves('run_id', 'sweep_YYYYMMDD_HHMMSS')

    args = parse_kv(varargin{:});
    if ~isfield(args,'latest'), args.latest = false; end
    if ~isfield(args,'run_id'), args.run_id = ''; end
    if ~isfield(args,'out_dir') || isempty(args.out_dir), args.out_dir = 'figures'; end
    if ~exist(args.out_dir,'dir'), mkdir(args.out_dir); end

    if args.latest
        run_id = find_latest_run_id('results','*_curves.mat');
        if isempty(run_id), error('No results/*_curves.mat found.'); end
    else
        run_id = char(args.run_id);
        if isempty(run_id), error('Please provide run_id or set latest=true.'); end
    end

    mat_path = fullfile('results',[run_id '_curves.mat']);
    S = load(mat_path);
    if ~isfield(S,'result'), error('MAT file missing variable "result".'); end
    result = S.result;

    x = result.x_vals(:);
    algs = result.algorithms;

    % Plot only Sum-Rate and QoE Cost
    if isfield(result,'sum_rate')
        clean_plot(run_id, args.out_dir, x, result.sum_rate, algs, ...
            'p (dBW)', 'Sum-rate', 'sum_rate');
    end
    if isfield(result,'avg_qoe')
        clean_plot(run_id, args.out_dir, x, result.avg_qoe, algs, ...
            'p (dBW)', 'Avg QoE Cost', 'avg_qoe');
    end

    fprintf('Figures saved to: %s\n', args.out_dir);
end

%% ========== CLEAN PLOT (MATLAB DEFAULT STYLE) ==========
function clean_plot(run_id, out_dir, x, Y, algs, xlab, ylab, fname)
    if isempty(Y), return; end

    figure('Visible', 'off');
    hold on;

    % Different markers for each algorithm (4 algorithms: random, norm, proposed, ga)
    markers = {'o', 's', 'd', '^'};  % circle, square, diamond, up-triangle
    
    for a = 1:numel(algs)
        marker = markers{mod(a-1, numel(markers)) + 1};
        plot(x, Y(:, a), ['-' marker], 'LineWidth', 1.5, 'MarkerSize', 6);
    end

    xlabel(xlab);
    ylabel(ylab);
    legend(algs, 'Location', 'best');
    grid on;

    % Save PNG
    png_path = fullfile(out_dir, [run_id '_' fname '.png']);
    saveas(gcf, png_path);
    close(gcf);
end

%% ========== HELPERS ==========
function args = parse_kv(varargin)
    args = struct();
    if isempty(varargin), return; end
    if mod(numel(varargin),2) ~= 0
        error('Arguments must be key-value pairs.');
    end
    for i = 1:2:numel(varargin)
        k = varargin{i}; v = varargin{i+1};
        if ~(ischar(k) || isstring(k))
            error('Key must be a string.');
        end
        args.(char(k)) = v;
    end
end

function run_id = find_latest_run_id(results_dir, pattern)
    d = dir(fullfile(results_dir, pattern));
    if isempty(d), run_id = ''; return; end
    [~, idx] = max([d.datenum]);
    name = d(idx).name;
    run_id = regexprep(name, '_curves\.mat$', '');
end
