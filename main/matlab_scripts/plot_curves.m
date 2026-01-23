function plot_curves(varargin)
%PLOT_CURVES Plot sweep results and save figures (old MATLAB compatible).
%
% Usage:
%   plot_curves('latest', true)
%   plot_curves('run_id', 'sweep_YYYYMMDD_HHMMSS')
%
% It reads results/<run_id>_curves.mat and writes figures/*.png

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
    mc = 1;
    if isfield(result,'mc') && ~isempty(result.mc), mc = result.mc; end

    % main metrics
    if isfield(result,'sum_rate')
        do_plot(run_id,args.out_dir,x,result.sum_rate, get_std(result,'sum_rate'), algs,mc,...
            'p (dBW)','Sum-rate','sum_rate');
    end
    if isfield(result,'avg_qoe')
        do_plot(run_id,args.out_dir,x,result.avg_qoe, get_std(result,'avg_qoe'), algs,mc,...
            'p (dBW)','Avg QoE cost (lower is better)','avg_qoe');
    end
    if isfield(result,'semantic_vio')
        do_plot(run_id,args.out_dir,x,result.semantic_vio, get_std(result,'semantic_vio'), algs,mc,...
            'p (dBW)','Semantic violation rate','semantic_vio');
    end
    if isfield(result,'deadline_vio')
        do_plot(run_id,args.out_dir,x,result.deadline_vio, get_std(result,'deadline_vio'), algs,mc,...
            'p (dBW)','Deadline violation rate','deadline_vio');
    end

    % fairness
    if isfield(result,'ris_usage')
        do_plot(run_id,args.out_dir,x,result.ris_usage, get_std(result,'ris_usage'), algs,mc,...
            'p (dBW)','RIS usage ratio','ris_usage');
    end
    if isfield(result,'unique_ris')
        do_plot(run_id,args.out_dir,x,result.unique_ris, get_std(result,'unique_ris'), algs,mc,...
            'p (dBW)','# Unique RIS used','unique_ris');
    end

    % runtime
    if isfield(result,'runtime_assign')
        do_plot(run_id,args.out_dir,x,result.runtime_assign, get_std(result,'runtime_assign'), algs,mc,...
            'p (dBW)','Runtime: assignment only (s)','runtime_assign');
    end
    if isfield(result,'runtime_total')
        do_plot(run_id,args.out_dir,x,result.runtime_total, get_std(result,'runtime_total'), algs,mc,...
            'p (dBW)','Runtime: total per algorithm (s)','runtime_total');
    end
end

% ---------------- helpers ----------------

function args = parse_kv(varargin)
    args = struct();
    if isempty(varargin), return; end
    if mod(numel(varargin),2) ~= 0
        error('Arguments must be key-value pairs.');
    end
    for i=1:2:numel(varargin)
        k = varargin{i}; v = varargin{i+1};
        if ~(ischar(k) || isstring(k))
            error('Key must be a string.');
        end
        args.(char(k)) = v;
    end
end

function s = get_std(result, base)
    fn = [base '_std'];
    if isfield(result, fn) && ~isempty(result.(fn))
        s = result.(fn);
    else
        s = [];
    end
end

function do_plot(run_id,out_dir,x,Y,Ystd,algs,mc,xlab,ylab,fname)
    if isempty(Y), return; end

    A = size(Y,2);

    figure('Visible','off'); hold on; grid on;

    use_err = ~isempty(Ystd);
    if use_err
        se = Ystd ./ max(1,sqrt(mc)); % standard error
    end

    for a=1:A
        y = Y(:,a);

        if use_err
            e = se(:,a);

            % 关键：不要在 errorbar 里用 name-value（旧版/某些实现会报参数不足）
            h = errorbar(x(:), y(:), e(:), '-o');

            % 线宽用 set 设置（兼容）
            try
                set(h,'LineWidth',1.2);
            catch
                % 某些环境 errorbar 返回非图元句柄，忽略
            end
        else
            h = plot(x(:), y(:), '-o');
            try
                set(h,'LineWidth',1.2);
            catch
            end
        end
    end

    xlabel(xlab); ylabel(ylab);
    legend(algs,'Location','best');
    try, set(gca,'FontSize',12); catch, end

    out_path = fullfile(out_dir,[run_id '_' fname '.png']);
    print(gcf, out_path, '-dpng', '-r200');
    close(gcf);
end

function run_id = find_latest_run_id(results_dir, pattern)
    d = dir(fullfile(results_dir, pattern));
    if isempty(d), run_id = ''; return; end
    [~, idx] = max([d.datenum]);
    name = d(idx).name;
    run_id = regexprep(name, '_curves\.mat$', '');
end
