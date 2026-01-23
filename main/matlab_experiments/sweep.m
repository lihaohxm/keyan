function run_id = sweep(varargin)
%SWEEP Sweep over p_dbw_list with MC trials. Compatible with current project APIs.
%
% Example:
%   run_id = sweep('mc',50,'p_dbw_list',[-10 -5 0], 'seed',1, ...
%                  'dmax',0.94,'beta_s',0.005,'b_s',50);
%   plot_curves('latest', true);

    addpath('matlab_sim', '-begin');
    addpath(fullfile('matlab_sim','matching'), '-begin');
    addpath('matlab_scripts', '-begin');

    cfg = config();
    cfg = apply_overrides(cfg, varargin{:});

    % defaults
    if ~isfield(cfg,'mc') || isempty(cfg.mc), cfg.mc = 100; end
    if ~isfield(cfg,'p_dbw_list') || isempty(cfg.p_dbw_list), cfg.p_dbw_list = [-10 -5 0]; end
    if ~isfield(cfg,'seed') || isempty(cfg.seed), cfg.seed = 1; end

    if ~isfield(cfg,'semantic_mode') || isempty(cfg.semantic_mode), cfg.semantic_mode = 'proxy'; end
    if ~isfield(cfg,'semantic_table'), cfg.semantic_table = ''; end
    if ~isfield(cfg,'proxy_a') || isempty(cfg.proxy_a), cfg.proxy_a = 0.6; end
    if ~isfield(cfg,'proxy_b') || isempty(cfg.proxy_b), cfg.proxy_b = 0.4; end
    if ~isfield(cfg,'weights') || isempty(cfg.weights), cfg.weights = [0.5 0.5]; end

    algorithms = {'random','norm','qoe','exhaustive','ga_placeholder'};
    A = numel(algorithms);

    x_vals = cfg.p_dbw_list(:).';
    X = numel(x_vals);
    mc = cfg.mc;

    fprintf('SWEEP overrides: dmax=%.4f, beta_s=%.4g, beta_d=%.4g, b_s=%.4g, b_d=%.4g\n', ...
        cfg.dmax, cfg.beta_s, cfg.beta_d, cfg.b_s, cfg.b_d);

    % output container (mean + std)
    result = struct();
    result.algorithms = algorithms;
    result.x_axis = 'p_dbw';
    result.x_vals = x_vals;
    result.mc = mc;

    % metrics (mean/std)
    result.sum_rate       = nan(X, A);
    result.sum_rate_std   = nan(X, A);

    result.avg_qoe        = nan(X, A);
    result.avg_qoe_std    = nan(X, A);

    result.deadline_vio     = nan(X, A);
    result.deadline_vio_std = nan(X, A);

    result.semantic_vio     = nan(X, A);
    result.semantic_vio_std = nan(X, A);

    result.ris_usage      = nan(X, A);
    result.ris_usage_std  = nan(X, A);

    result.unique_ris      = nan(X, A);
    result.unique_ris_std  = nan(X, A);

    result.runtime_assign      = nan(X, A);
    result.runtime_assign_std  = nan(X, A);

    result.runtime_total      = nan(X, A);
    result.runtime_total_std  = nan(X, A);

    % sweep loop
    for ix = 1:X
        p_dbw = x_vals(ix);

        sr_mc = zeros(mc, A);
        q_mc  = zeros(mc, A);
        dv_mc = zeros(mc, A);
        sv_mc = zeros(mc, A);

        usage_mc = zeros(mc, A);
        uniq_mc  = zeros(mc, A);

        ta_mc = zeros(mc, A);
        tt_mc = zeros(mc, A);

        for tr = 1:mc
            trial_seed = cfg.seed + (ix-1)*100000 + tr - 1;

            geom = geometry(cfg, trial_seed);
            ch   = channel(cfg, geom, trial_seed);

            for a = 1:A
                alg = algorithms{a};

                t0 = tic;

                t_assign0 = tic;
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed);
                ta_mc(tr,a) = toc(t_assign0);

                h_eff = effective_channel(cfg, ch, assign);

                [gamma, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);

                xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
                    struct('a', cfg.proxy_a, 'b', cfg.proxy_b));

                [avg_qoe, ~, meta] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:));

                sr_mc(tr,a) = sum_rate;
                q_mc(tr,a)  = avg_qoe;

                dv_mc(tr,a) = 1 - meta.deadline_ok;
                sv_mc(tr,a) = 1 - meta.semantic_ok;

                [usage_mc(tr,a), uniq_mc(tr,a)] = ris_stats(assign);

                tt_mc(tr,a) = toc(t0);
            end
        end

        result.sum_rate(ix,:)     = mean(sr_mc, 1);
        result.sum_rate_std(ix,:) = std(sr_mc, 0, 1);

        result.avg_qoe(ix,:)      = mean(q_mc, 1);
        result.avg_qoe_std(ix,:)  = std(q_mc, 0, 1);

        result.deadline_vio(ix,:)     = mean(dv_mc, 1);
        result.deadline_vio_std(ix,:) = std(dv_mc, 0, 1);

        result.semantic_vio(ix,:)     = mean(sv_mc, 1);
        result.semantic_vio_std(ix,:) = std(sv_mc, 0, 1);

        result.ris_usage(ix,:)     = mean(usage_mc, 1);
        result.ris_usage_std(ix,:) = std(usage_mc, 0, 1);

        result.unique_ris(ix,:)     = mean(uniq_mc, 1);
        result.unique_ris_std(ix,:) = std(uniq_mc, 0, 1);

        result.runtime_assign(ix,:)     = mean(ta_mc, 1);
        result.runtime_assign_std(ix,:) = std(ta_mc, 0, 1);

        result.runtime_total(ix,:)     = mean(tt_mc, 1);
        result.runtime_total_std(ix,:) = std(tt_mc, 0, 1);

        fprintf('x=%g done. mean(avgQ)=%s ; mean(sv)=%s\n', ...
            p_dbw, mat2str(result.avg_qoe(ix,:),6), mat2str(result.semantic_vio(ix,:),6));
    end

    % save (your save_results generates run_id)
    run_id = save_results('sweep', result, x_vals);

    % plot latest safely
    try
        clear plot_curves; rehash;
        plot_curves('latest', true);
    catch ME
        warning('plot_curves failed: %s', ME.message);
    end

    fprintf('Sweep complete. Results saved with run_id: %s\n', run_id);
end

% ---------------- helpers ----------------

function cfg = apply_overrides(cfg, varargin)
    if mod(numel(varargin),2) ~= 0
        error('Overrides must be key-value pairs.');
    end
    for i = 1:2:numel(varargin)
        k = char(varargin{i});
        v = varargin{i+1};
        cfg.(k) = v; % allow new fields like seed
    end
end

function assign = pick_assignment(cfg, ch, alg, p_dbw, seed)
    if isfield(cfg,'weights') && ~isempty(cfg.weights)
        w = cfg.weights(1,:);
    else
        w = [0.5 0.5];
    end
    semantic_mode = cfg.semantic_mode;
    table_path = cfg.semantic_table;

    switch alg
        case 'random'
            assign = random_match(cfg, seed);
        case 'norm'
            assign = norm_based(cfg, ch);
        case 'qoe'
            assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, w);
        case 'exhaustive'
            [assign, info] = exhaustive(cfg, ch, p_dbw);
            if isfield(info,'skipped') && info.skipped
                assign = norm_based(cfg, ch);
            end
        case 'ga_placeholder'
            assign = ga_placeholder(cfg);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

function [usage, n_unique] = ris_stats(assign)
    if isempty(assign)
        usage = 0; n_unique = 0; return;
    end
    pos = assign(assign > 0);
    if isempty(pos)
        n_unique = 0;
    else
        n_unique = numel(unique(pos(:)));
    end
    usage = mean(assign(:) > 0); % your project uses vector assignment
end
