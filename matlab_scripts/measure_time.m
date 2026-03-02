function measure_time()
%MEASURE_TIME 测量各算法的运行时间

    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    proj_root = fileparts(this_dir);
    
    addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
    addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
    
    old_dir = cd(proj_root);
    
    fprintf('========================================\n');
    fprintf('算法运行时间测量\n');
    fprintf('========================================\n\n');
    
    % 测试配置
    mc = 100;  % 测量次数
    p_dbw = -15;
    seed = 42;
    
    cfg = config();
    cfg.users_per_cell = 12;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 12;
    cfg.num_ris = 4;
    
    fprintf('测试配置: K=%d, L=%d, MC=%d\n\n', cfg.num_users, cfg.num_ris, mc);
    
    % 预分配
    time_random = zeros(mc, 1);
    time_norm = zeros(mc, 1);
    time_proposed = zeros(mc, 1);
    time_ga = zeros(mc, 1);
    
    fprintf('开始测量...\n');
    
    for tr = 1:mc
        trial_seed = seed + tr;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        
        % Random
        tic;
        assign_rand = random_match(cfg, trial_seed);
        time_random(tr) = toc * 1000;
        
        % Norm-based
        tic;
        assign_norm = norm_based(cfg, ch);
        time_norm(tr) = toc * 1000;
        
        % Proposed (QoE-aware)
        tic;
        assign_proposed = qoe_aware(cfg, ch, p_dbw, cfg.semantic_mode, ...
            cfg.semantic_table, cfg.weights(1,:), geom);
        time_proposed(tr) = toc * 1000;
        
        % GA
        opts.verbose = false;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.weights = cfg.weights(1,:);
        opts.geom = geom;
        opts.optimize_sumrate = false;
        opts.Np = cfg.ga_Np;
        opts.Niter = cfg.ga_Niter;
        
        tic;
        [assign_ga, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        time_ga(tr) = toc * 1000;
        
        if mod(tr, 20) == 0
            fprintf('  进度: %d/%d\n', tr, mc);
        end
    end
    
    fprintf('\n========================================\n');
    fprintf('测量结果 (单位: ms)\n');
    fprintf('========================================\n\n');
    
    fprintf('%-12s %10s %10s %10s %10s\n', '算法', '平均', '标准差', '最小', '最大');
    fprintf('%s\n', repmat('-', 1, 54));
    fprintf('%-12s %10.3f %10.3f %10.3f %10.3f\n', 'Random', mean(time_random), std(time_random), min(time_random), max(time_random));
    fprintf('%-12s %10.3f %10.3f %10.3f %10.3f\n', 'Norm', mean(time_norm), std(time_norm), min(time_norm), max(time_norm));
    fprintf('%-12s %10.3f %10.3f %10.3f %10.3f\n', 'Proposed', mean(time_proposed), std(time_proposed), min(time_proposed), max(time_proposed));
    fprintf('%-12s %10.3f %10.3f %10.3f %10.3f\n', 'GA', mean(time_ga), std(time_ga), min(time_ga), max(time_ga));
    
    fprintf('\n========================================\n');
    fprintf('相对时间 (以Random为基准)\n');
    fprintf('========================================\n\n');
    
    base_time = mean(time_random);
    fprintf('Random:   %.1fx (基准)\n', 1.0);
    fprintf('Norm:     %.1fx\n', mean(time_norm) / base_time);
    fprintf('Proposed: %.1fx\n', mean(time_proposed) / base_time);
    fprintf('GA:       %.1fx\n', mean(time_ga) / base_time);
    
    fprintf('\n========================================\n');
    fprintf('GA参数: Np=%d, Niter=%d\n', cfg.ga_Np, cfg.ga_Niter);
    fprintf('========================================\n');
    
    % 测试不同GA配置
    fprintf('\n========================================\n');
    fprintf('GA不同配置的时间\n');
    fprintf('========================================\n\n');
    
    ga_configs = [
        1, 5;
        1, 10;
        2, 10;
        5, 20;
        10, 30;
        20, 50;
    ];
    
    mc_ga = 20;  % GA测试次数
    
    fprintf('%-8s %-8s %12s\n', 'Niter', 'Np', '平均时间(ms)');
    fprintf('%s\n', repmat('-', 1, 30));
    
    for g = 1:size(ga_configs, 1)
        niter = ga_configs(g, 1);
        np = ga_configs(g, 2);
        
        tmp_time = zeros(mc_ga, 1);
        
        for tr = 1:mc_ga
            trial_seed = seed + tr;
            geom = geometry(cfg, trial_seed);
            ch = channel(cfg, geom, trial_seed);
            
            opts.Niter = niter;
            opts.Np = np;
            
            tic;
            [~, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
            tmp_time(tr) = toc * 1000;
        end
        
        fprintf('%-8d %-8d %12.2f\n', niter, np, mean(tmp_time));
    end
    
    cd(old_dir);
end
