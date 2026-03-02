function run_id = sweep_advanced(experiment_type, varargin)
%SWEEP_ADVANCED Advanced experiments with consistent channel per MC trial.

    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    proj_root = fileparts(this_dir);
    
    addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
    addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
    addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');
    
    old_dir = cd(proj_root);

    p = inputParser;
    addParameter(p, 'mc', 200);
    addParameter(p, 'seed', 42);
    parse(p, varargin{:});
    
    mc = p.Results.mc;
    base_seed = p.Results.seed;

    try
        switch lower(experiment_type)
            case 'heterogeneous'
                run_id = exp_heterogeneous(mc, base_seed, proj_root);
            case 'scarcity'
                run_id = exp_scarcity(mc, base_seed, proj_root);
            case 'constraint'
                run_id = exp_constraint(mc, base_seed, proj_root);
            case 'ris_count'
                run_id = exp_ris_count(mc, base_seed, proj_root);
            otherwise
                error('Unknown experiment type: %s', experiment_type);
        end
    catch ME
        cd(old_dir);
        rethrow(ME);
    end
    
    cd(old_dir);
end

%% Heterogeneous Users - sweep power with same channel per trial
function run_id = exp_heterogeneous(mc, base_seed, proj_root)
    fprintf('\n=== Experiment: Heterogeneous Users ===\n');
    
    cfg = config();
    cfg.users_per_cell = 6;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 6;
    cfg.num_ris = 4;
    
    % 閻劍鍩涢弶鍐櫢: URLLC, Semantic, Best Effort
    user_weights = [
        0.8, 0.2;  % User 1-2: URLLC (delay-sensitive)
        0.8, 0.2;
        0.2, 0.8;  % User 3-4: Semantic
        0.2, 0.8;
        0.5, 0.5;  % User 5-6: Best Effort
        0.5, 0.5;
    ];
    
    algorithms = {'random', 'norm', 'proposed', 'ga'};
    p_dbw_list = linspace(-25, -5, 8);
    
    A = numel(algorithms);
    X = numel(p_dbw_list);
    
    q_all = zeros(mc, X, A);
    sr_all = zeros(mc, X, A);
    urllc_all = zeros(mc, X, A);
    semantic_all = zeros(mc, X, A);
    besteffort_all = zeros(mc, X, A);
    
    for tr = 1:mc
        trial_seed = base_seed + tr;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        
        for ix = 1:X
            p_dbw = p_dbw_list(ix);
            
            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);
                
                out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, user_weights);
                qoe_vec = out.qoe_vec;
                sum_rate = out.sum_rate_bps;
                
                q_all(tr, ix, a) = mean(qoe_vec);
                sr_all(tr, ix, a) = sum_rate;
                urllc_all(tr, ix, a) = mean(qoe_vec(1:2));      % 閻劍鍩?-2: URLLC
                semantic_all(tr, ix, a) = mean(qoe_vec(3:4));   % 閻劍鍩?-4: Semantic
                besteffort_all(tr, ix, a) = mean(qoe_vec(5:6)); % 閻劍鍩?-6: Best Effort
            end
        end
        
        if mod(tr, 50) == 0
            fprintf('  MC trial %d/%d\n', tr, mc);
        end
    end
    
    result = struct();
    result.x_vals = p_dbw_list;
    result.algorithms = algorithms;
    result.sum_rate = squeeze(mean(sr_all, 1));
    result.avg_qoe = squeeze(mean(q_all, 1));
    result.urllc_qoe = squeeze(mean(urllc_all, 1));
    result.semantic_qoe = squeeze(mean(semantic_all, 1));
    result.besteffort_qoe = squeeze(mean(besteffort_all, 1));
    result.experiment = 'heterogeneous';
    
    run_id = save_results('hetero', result, p_dbw_list);
    plot_hetero_results(run_id, result, proj_root);
end

%% Resource Scarcity - different K values
function run_id = exp_scarcity(mc, base_seed, proj_root)
    fprintf('\n=== Experiment: Resource Scarcity ===\n');

    k_values = [3, 4, 5, 6, 7, 8];
    L = 4;
    k0 = 1;
    p_dbw = -15;

    algorithms = {'random', 'norm', 'proposed', 'ga'};
    A = numel(algorithms);
    X = numel(k_values);

    result = struct();
    result.x_vals = k_values;
    result.algorithms = algorithms;
    result.sum_rate = zeros(X, A);
    result.avg_qoe = zeros(X, A);

    for ix = 1:X
        K = k_values(ix);
        fprintf('  K=%d\n', K);

        cfg = config();
        cfg.users_per_cell = K;
        cfg.ris_per_cell = L;
        cfg.k0 = k0;
        cfg.num_users = K;
        cfg.num_ris = L;

        sr_mc = zeros(mc, A);
        qoe_mc = zeros(mc, A);

        for tr = 1:mc
            trial_seed = base_seed + tr;
            geom = geometry(cfg, trial_seed);
            ch = channel(cfg, geom, trial_seed);

            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);
                out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
                sr_mc(tr, a) = out.sum_rate_bps;
                qoe_mc(tr, a) = out.avg_qoe;
            end
        end

        result.sum_rate(ix, :) = mean(sr_mc, 1);
        result.avg_qoe(ix, :) = mean(qoe_mc, 1);
    end

    result.experiment = 'scarcity';
    run_id = save_results('scarcity', result, k_values);
    plot_scarcity_results(run_id, result, proj_root);
end
%% Constraint Satisfaction
function run_id = exp_constraint(mc, base_seed, proj_root)
    fprintf('\n=== Experiment: Constraint Satisfaction ===\n');

    cfg = config();
    cfg.users_per_cell = 6;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 6;
    cfg.num_ris = 4;

    algorithms = {'random', 'norm', 'proposed', 'ga'};
    p_dbw_list = linspace(-25, -5, 8);

    A = numel(algorithms);
    X = numel(p_dbw_list);

    q_all = zeros(mc, X, A);
    sr_all = zeros(mc, X, A);

    for tr = 1:mc
        trial_seed = base_seed + tr;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);

        for ix = 1:X
            p_dbw = p_dbw_list(ix);
            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);
                out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
                q_all(tr, ix, a) = out.avg_qoe;
                sr_all(tr, ix, a) = out.sum_rate_bps;
            end
        end

        if mod(tr, 50) == 0
            fprintf('  MC trial %d/%d\n', tr, mc);
        end
    end

    result = struct();
    result.x_vals = p_dbw_list;
    result.algorithms = algorithms;
    result.sum_rate = squeeze(mean(sr_all, 1));
    result.avg_qoe = squeeze(mean(q_all, 1));
    result.experiment = 'constraint';

    run_id = save_results('constraint', result, p_dbw_list);
    plot_constraint_results(run_id, result, proj_root);
end
%% RIS Count Sweep
function run_id = exp_ris_count(mc, base_seed, proj_root)
    fprintf('\n=== Experiment: RIS Count Sweep ===\n');

    L_values = [2, 3, 4, 5, 6, 7, 8, 10];
    K = 6;
    k0 = 1;
    p_dbw = -15;

    algorithms = {'random', 'norm', 'proposed', 'ga'};
    A = numel(algorithms);
    X = numel(L_values);

    result = struct();
    result.x_vals = L_values;
    result.algorithms = algorithms;
    result.sum_rate = zeros(X, A);
    result.avg_qoe = zeros(X, A);

    for ix = 1:X
        L = L_values(ix);
        fprintf('  L=%d\n', L);

        cfg = config();
        cfg.users_per_cell = K;
        cfg.ris_per_cell = L;
        cfg.k0 = k0;
        cfg.num_users = K;
        cfg.num_ris = L;

        sr_mc = zeros(mc, A);
        qoe_mc = zeros(mc, A);

        for tr = 1:mc
            trial_seed = base_seed + tr;
            geom = geometry(cfg, trial_seed);
            ch = channel(cfg, geom, trial_seed);

            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);
                out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
                sr_mc(tr, a) = out.sum_rate_bps;
                qoe_mc(tr, a) = out.avg_qoe;
            end
        end

        result.sum_rate(ix, :) = mean(sr_mc, 1);
        result.avg_qoe(ix, :) = mean(qoe_mc, 1);
    end

    result.experiment = 'ris_count';
    run_id = save_results('ris_count', result, L_values);
    plot_ris_count_results(run_id, result, proj_root);
end
%% Helper Functions

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

function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    if size(weights, 1) == 1
        profile.weights = repmat(weights(:).', cfg.num_users, 1);
    else
        profile.weights = weights;
    end
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
end

function plot_hetero_results(run_id, result, proj_root)
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    x = result.x_vals;
    algs = result.algorithms;
    A = numel(algs);
    markers = {'o', 's', 'd', '^'};
    
    % Avg QoE
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.avg_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Avg QoE Cost');
    title('Heterogeneous Users: Avg QoE Cost');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('hetero_%s_avg_qoe.png', run_id)));
    close(gcf);
    
    % URLLC QoE
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.urllc_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('URLLC Users QoE Cost');
    title('Heterogeneous: URLLC Users (w_d=0.8, w_s=0.2)');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('hetero_%s_urllc_qoe.png', run_id)));
    close(gcf);
    
    % Semantic QoE (婵″倹鐏夌€涙ê婀?
    if isfield(result, 'semantic_qoe')
        figure('Position', [100 100 700 500]);
        hold on;
        for a = 1:A
            plot(x, result.semantic_qoe(:, a), ['-' markers{a}], ...
                'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
        end
        hold off;
        xlabel('p (dBW)');
        ylabel('Semantic Users QoE Cost');
        title('Heterogeneous: Semantic Users (w_d=0.2, w_s=0.8)');
        legend('Location', 'best');
        grid on;
        saveas(gcf, fullfile(out_dir, sprintf('hetero_%s_semantic_qoe.png', run_id)));
        close(gcf);
    end
    
    % Best Effort QoE 娑撳秴鍟€閻㈢喐鍨?
    
    fprintf('Saved: hetero_%s_avg_qoe.png, hetero_%s_urllc_qoe.png\n', run_id, run_id);
    if isfield(result, 'semantic_qoe')
        fprintf('       hetero_%s_semantic_qoe.png\n', run_id);
    end
end

function plot_scarcity_results(run_id, result, proj_root)
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    x = result.x_vals;
    algs = result.algorithms;
    A = numel(algs);
    markers = {'o', 's', 'd', '^'};
    
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.avg_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
    end
    hold off;
    xlabel('Number of Users (K)');
    ylabel('Avg QoE Cost');
    title('Resource Scarcity: Avg QoE Cost');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('scarcity_%s_avg_qoe.png', run_id)));
    close(gcf);
    
    fprintf('Saved: scarcity_%s_avg_qoe.png\n', run_id);
end

function plot_constraint_results(run_id, result, proj_root)
    % Constraint鐎圭偤鐛欐稉宥呭晙閻㈢喐鍨歛vg_qoe閸?
    % 鐠囥儱鐤勬宀€娈戠紒鎾寸亯瀹告彃瀵橀崥顐㈡躬閸忔湹绮崶鍙ヨ厬
end

function plot_ris_count_results(run_id, result, proj_root)
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    x = result.x_vals;
    algs = result.algorithms;
    A = numel(algs);
    markers = {'o', 's', 'd', '^'};
    
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.avg_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'DisplayName', algs{a});
    end
    hold off;
    xlabel('Number of RIS (L)');
    ylabel('Avg QoE Cost');
    title('RIS Count: Avg QoE Cost');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('ris_count_%s_avg_qoe.png', run_id)));
    close(gcf);
    
    fprintf('Saved: ris_count_%s_avg_qoe.png\n', run_id);
end



