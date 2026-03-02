function plot_ga_convergence(varargin)
%PLOT_GA_CONVERGENCE 鏃堕棿-鎬ц兘鏉冭　鍥?
%
% 灞曠ず鏍稿績璁虹偣锛?
%   - Proposed 鐢ㄦ椂鏋佺煭锛屾€ц兘鎺ヨ繎鏈€浼?
%   - GA 闇€瑕?N 鍊嶆椂闂存墠鑳借揪鍒?瓒呰秺 Proposed
%   - 鍦ㄥ疄鏃剁害鏉熶笅锛孭roposed 鏄洿濂界殑閫夋嫨

    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    proj_root = fileparts(this_dir);
    
    addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
    addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
    
    p = inputParser;
    addParameter(p, 'mc', 50);
    addParameter(p, 'p_dbw', -15);
    addParameter(p, 'save', true);
    parse(p, varargin{:});
    
    mc = p.Results.mc;
    p_dbw = p.Results.p_dbw;
    
    fprintf('========================================\n');
    fprintf('Time-Performance Trade-off Analysis\n');
    fprintf('========================================\n\n');
    
    cfg = config();
    % 浣跨敤榛樿瑙勬ā K=12, L=4
    cfg.users_per_cell = 12;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 12;
    cfg.num_ris = 4;
    
    seed = 42;
    
    fprintf('闂瑙勬ā: K=%d users, L=%d RIS\n\n', cfg.num_users, cfg.num_ris);
    
    % GA configs: [Niter, Np] - 浠庡皬鍒板ぇ
    ga_configs = [
        1, 10;     % 鏈€蹇?
        2, 15;
        5, 20;
        10, 30;
        20, 50;
        50, 80;
        100, 100;  % 鏈€鎱絾鏈€浼?
    ];
    
    num_configs = size(ga_configs, 1);
    
    ga_qoe = zeros(num_configs, 1);
    ga_time = zeros(num_configs, 1);
    proposed_qoe = 0;
    proposed_time = 0;
    random_qoe = 0;
    norm_qoe = 0;
    
    % ========== Test Proposed ==========
    fprintf('娴嬭瘯 Proposed...\n');
    tmp_qoe = zeros(mc, 1);
    tmp_time = zeros(mc, 1);
    
    for tr = 1:mc
        trial_seed = seed + tr;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        
        tic;
        assign = qoe_aware(cfg, ch, p_dbw, cfg.semantic_mode, cfg.semantic_table, ...
            cfg.weights(1,:), geom);
        tmp_time(tr) = toc * 1000;
        
        [tmp_qoe(tr), ~] = evaluate_assignment(cfg, ch, assign, geom, p_dbw);
    end
    
    proposed_qoe = mean(tmp_qoe);
    proposed_time = mean(tmp_time);
    fprintf('  Proposed: QoE=%.4f, Time=%.3fms\n', proposed_qoe, proposed_time);
    
    % ========== Test Random & Norm (baselines) ==========
    fprintf('娴嬭瘯 Baselines...\n');
    tmp_rand = zeros(mc, 1);
    tmp_norm = zeros(mc, 1);
    
    for tr = 1:mc
        trial_seed = seed + tr;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        
        assign_rand = random_match(cfg, trial_seed);
        assign_norm = norm_based(cfg, ch);
        
        [tmp_rand(tr), ~] = evaluate_assignment(cfg, ch, assign_rand, geom, p_dbw);
        [tmp_norm(tr), ~] = evaluate_assignment(cfg, ch, assign_norm, geom, p_dbw);
    end
    
    random_qoe = mean(tmp_rand);
    norm_qoe = mean(tmp_norm);
    fprintf('  Random: QoE=%.4f\n', random_qoe);
    fprintf('  Norm:   QoE=%.4f\n', norm_qoe);
    
    % ========== Test GA with different configs ==========
    fprintf('\n娴嬭瘯 GA 涓嶅悓閰嶇疆...\n');
    
    for g = 1:num_configs
        niter = ga_configs(g, 1);
        np = ga_configs(g, 2);
        
        tmp_qoe = zeros(mc, 1);
        tmp_time = zeros(mc, 1);
        
        for tr = 1:mc
            trial_seed = seed + tr;
            geom = geometry(cfg, trial_seed);
            ch = channel(cfg, geom, trial_seed);
            
            opts.verbose = false;
            opts.semantic_mode = cfg.semantic_mode;
            opts.table_path = cfg.semantic_table;
            opts.weights = cfg.weights(1,:);
            opts.geom = geom;
            opts.optimize_sumrate = false;
            opts.Niter = niter;
            opts.Np = np;
            
            tic;
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
            tmp_time(tr) = toc * 1000;
            
            [tmp_qoe(tr), ~] = evaluate_assignment(cfg, ch, assign, geom, p_dbw);
        end
        
        ga_qoe(g) = mean(tmp_qoe);
        ga_time(g) = mean(tmp_time);
        
        fprintf('  Niter=%2d, Np=%2d: QoE=%.4f, Time=%.2fms\n', ...
            niter, np, ga_qoe(g), ga_time(g));
    end
    
    % ========== Plot ==========
    figure('Position', [100 100 700 500]);
    
    % GA鏇茬嚎
    plot(ga_time, ga_qoe, '-s', 'LineWidth', 2, 'MarkerSize', 10, ...
        'MarkerFaceColor', [0.3 0.6 0.9], 'Color', [0.2 0.4 0.8], ...
        'DisplayName', 'GA');
    hold on;
    
    % Proposed鐐?
    plot(proposed_time, proposed_qoe, 'p', 'MarkerSize', 18, ...
        'MarkerFaceColor', [0.9 0.6 0.1], 'MarkerEdgeColor', [0.7 0.4 0], ...
        'LineWidth', 2, 'DisplayName', 'Proposed');
    
    % 姘村钩鍙傝€冪嚎
    xl = xlim;
    xl(1) = 0.1;
    xl(2) = max(ga_time) * 1.5;
    
    plot(xl, [random_qoe random_qoe], ':', 'LineWidth', 1.5, ...
        'Color', [0.7 0.7 0.7], 'DisplayName', 'Random');
    plot(xl, [norm_qoe norm_qoe], '--', 'LineWidth', 1.5, ...
        'Color', [0.5 0.5 0.5], 'DisplayName', 'Norm');
    
    hold off;
    
    set(gca, 'XScale', 'log');
    xlim(xl);
    xlabel('Runtime (ms)', 'FontSize', 12);
    ylabel('Average QoE Cost (lower is better)', 'FontSize', 12);
    title('Time-Performance Trade-off', 'FontSize', 14);
    legend('Location', 'northeast', 'FontSize', 10);
    grid on;
    
    % ========== Key findings ==========
    fprintf('\n========================================\n');
    fprintf('鍏抽敭鍙戠幇\n');
    fprintf('========================================\n');
    
    % 鎵惧埌GA棣栨杈惧埌proposed鎬ц兘鐨勭偣
    match_idx = find(ga_qoe <= proposed_qoe, 1);
    if ~isempty(match_idx)
        fprintf('GA 棣栨杈惧埌 Proposed 鎬ц兘:\n');
        fprintf('  閰嶇疆: Niter=%d, Np=%d\n', ga_configs(match_idx,1), ga_configs(match_idx,2));
        fprintf('  GA鏃堕棿: %.2f ms\n', ga_time(match_idx));
        fprintf('  Proposed鏃堕棿: %.3f ms\n', proposed_time);
        speedup = ga_time(match_idx)/proposed_time;
        fprintf('  鈫?Proposed 蹇?%.1f 鍊?\n', speedup);
    else
        fprintf('鍦ㄦ祴璇曡寖鍥村唴锛孏A鏈兘杈惧埌Proposed鎬ц兘\n');
        fprintf('鏈€鎺ヨ繎鐨凣A閰嶇疆: Niter=%d, Np=%d, QoE=%.4f\n', ...
            ga_configs(end,1), ga_configs(end,2), ga_qoe(end));
        speedup = ga_time(1)/proposed_time;
        fprintf('鍗充娇鏈€蹇獹A閰嶇疆锛孭roposed浠嶅揩 %.1f 鍊峔n', speedup);
    end
    
    % 鎬ц兘鎻愬崌
    fprintf('\n鐩告瘮 Random: Proposed 闄嶄綆 %.1f%% QoE Cost\n', ...
        (random_qoe - proposed_qoe) / random_qoe * 100);
    fprintf('鐩告瘮 Norm:   Proposed 闄嶄綆 %.1f%% QoE Cost\n', ...
        (norm_qoe - proposed_qoe) / norm_qoe * 100);
    
    % 鍏抽敭浼樺娍鎬荤粨
    fprintf('\n========================================\n');
    fprintf('Proposed绠楁硶鐨勬牳蹇冧紭鍔縗n');
    fprintf('========================================\n');
    fprintf('1. 纭畾鎬х粨鏋?- GA姣忔杩愯缁撴灉涓嶅悓\n');
    fprintf('2. 鐞嗚鍙垎鏋?- 鍙瘉鏄庤繎浼兼瘮\n');
    fprintf('3. 浣庡鏉傚害 - O(K虏L) vs O(Np路Niter路KL)\n');
    fprintf('4. 瀹炴椂鍙 - 婊¤冻姣绾у喅绛栭渶姹俓n');
    
    % ========== Save ==========
    if p.Results.save
        out_dir = fullfile(proj_root, 'figures');
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        
        filename = sprintf('ga_convergence_%s', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, fullfile(out_dir, [filename '.png']));
        fprintf('\n鍥剧墖宸蹭繚瀛? %s.png\n', filename);
    end
end

function [qoe_val, sum_rate] = evaluate_assignment(cfg, ch, assign, geom, p_dbw)
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, [], struct());
    qoe_val = out.avg_qoe;
    sum_rate = out.sum_rate_bps;
end
