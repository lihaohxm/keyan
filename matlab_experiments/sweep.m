function run_id = sweep(varargin)
%SWEEP Sweep over p_dbw_list with MC trials.
%
% 所有功率点使用相同的信道实现，保证曲线平滑

    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    proj_root = fileparts(this_dir);
    
    addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
    addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
    addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');
    
    old_dir = cd(proj_root);

    cfg = config();
    cfg = apply_overrides(cfg, varargin{:});

    if ~isfield(cfg,'mc') || isempty(cfg.mc), cfg.mc = 200; end
    if ~isfield(cfg,'seed') || isempty(cfg.seed), cfg.seed = 42; end
    
    cfg.p_dbw_list = linspace(-25, -5, 8);

    algorithms = {'random', 'norm', 'proposed', 'ga'};
    A = numel(algorithms);

    x_vals = cfg.p_dbw_list(:).';
    X = numel(x_vals);
    mc = cfg.mc;

    fprintf('========================================\n');
    fprintf('SWEEP Experiment\n');
    fprintf('========================================\n');
    fprintf('Power: [%.0f, %.0f] dBW, MC=%d\n', x_vals(1), x_vals(end), mc);
    fprintf('Algorithms: %s\n\n', strjoin(algorithms, ', '));

    result = struct();
    result.algorithms = algorithms;
    result.x_axis = 'p_dbw';
    result.x_vals = x_vals;
    result.mc = mc;

    result.sum_rate = nan(X, A);
    result.avg_qoe = nan(X, A);

    % 按MC trial外层循环，保证同一trial的所有功率点用相同信道
    sr_all = zeros(mc, X, A);
    q_all = zeros(mc, X, A);

    for tr = 1:mc
        % 每个trial使用固定种子生成信道
        trial_seed = cfg.seed + tr;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);

        for ix = 1:X
            p_dbw = x_vals(ix);

            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);

                h_eff = effective_channel(cfg, ch, assign);
                [gamma, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);

                xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
                    struct('a', cfg.proxy_a, 'b', cfg.proxy_b));

                prop_delay = calc_prop_delay(assign, geom);
                [avg_qoe, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:), prop_delay);

                sr_all(tr, ix, a) = sum_rate;
                q_all(tr, ix, a) = avg_qoe;
            end
        end

        if mod(tr, 50) == 0
            fprintf('  MC trial %d/%d\n', tr, mc);
        end
    end

    result.sum_rate = squeeze(mean(sr_all, 1));
    result.avg_qoe = squeeze(mean(q_all, 1));

    for ix = 1:X
        fprintf('P=%g dBW: QoE=%s\n', x_vals(ix), mat2str(result.avg_qoe(ix,:), 4));
    end

    if ~exist('results', 'dir'), mkdir('results'); end
    if ~exist('figures', 'dir'), mkdir('figures'); end
    
    run_id = save_results('sweep', result, x_vals);

    try
        plot_sweep_results(run_id, result, proj_root);
    catch ME
        warning('Plot failed: %s', ME.message);
    end

    fprintf('\nSweep complete. run_id: %s\n', run_id);
    cd(old_dir);
end

function cfg = apply_overrides(cfg, varargin)
    if mod(numel(varargin),2) ~= 0
        error('Overrides must be key-value pairs.');
    end
    for i = 1:2:numel(varargin)
        k = char(varargin{i});
        v = varargin{i+1};
        cfg.(k) = v;
    end
end

function assign = pick_assignment(cfg, ch, alg, p_dbw, seed, geom)
    if nargin < 6, geom = []; end
    
    w = cfg.weights(1,:);
    semantic_mode = cfg.semantic_mode;
    table_path = cfg.semantic_table;

    switch alg
        case 'random'
            assign = random_match(cfg, seed);
        case 'norm'
            assign = norm_based(cfg, ch);
        case 'proposed'
            assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, w, geom);
        case 'ga'
            opts.verbose = false;
            opts.semantic_mode = semantic_mode;
            opts.table_path = table_path;
            opts.weights = w;
            opts.geom = geom;
            opts.optimize_sumrate = false;
            opts.Np = cfg.ga_Np;
            opts.Niter = cfg.ga_Niter;
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

function prop_delay = calc_prop_delay(assign, geom)
    num_users = numel(assign);
    prop_delay = zeros(num_users, 1);
    prop_delay_factor = 1e-5;
    
    if isempty(geom) || ~isfield(geom, 'ris') || ~isfield(geom, 'ue')
        return;
    end
    
    bs_pos = geom.bs(1, :);
    
    for k = 1:num_users
        l = assign(k);
        if l > 0 && l <= size(geom.ris, 1)
            ris_pos = geom.ris(l, :);
            ue_pos = geom.ue(k, :);
            d_bs_ris = norm(ris_pos - bs_pos);
            d_ris_ue = norm(ue_pos - ris_pos);
            prop_delay(k) = prop_delay_factor * (d_bs_ris + d_ris_ue);
        end
    end
end

function plot_sweep_results(run_id, result, proj_root)
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    x = result.x_vals;
    algs = result.algorithms;
    A = numel(algs);
    markers = {'o', 's', 'd', '^'};
    
    % Avg QoE Cost
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.avg_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Avg QoE Cost');
    legend('Location', 'best');
    grid on;
    
    saveas(gcf, fullfile(out_dir, sprintf('sweep_%s_avg_qoe.png', run_id)));
    close(gcf);
    
    % Sum-rate
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.sum_rate(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Sum-rate');
    legend('Location', 'best');
    grid on;
    
    saveas(gcf, fullfile(out_dir, sprintf('sweep_%s_sum_rate.png', run_id)));
    close(gcf);
    
    fprintf('Saved: sweep_%s_avg_qoe.png, sweep_%s_sum_rate.png\n', run_id, run_id);
end
