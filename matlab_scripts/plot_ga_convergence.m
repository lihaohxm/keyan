function plot_ga_convergence(varargin)
%PLOT_GA_CONVERGENCE 时间-性能权衡图
%
% 展示核心论点：
%   - Proposed 用时极短，性能接近最优
%   - GA 需要 N 倍时间才能达到/超越 Proposed
%   - 在实时约束下，Proposed 是更好的选择

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
    % 使用默认规模 K=12, L=4
    cfg.users_per_cell = 12;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 12;
    cfg.num_ris = 4;
    
    seed = 42;
    
    fprintf('问题规模: K=%d users, L=%d RIS\n\n', cfg.num_users, cfg.num_ris);
    
    % GA configs: [Niter, Np] - 从小到大
    ga_configs = [
        1, 10;     % 最快
        2, 15;
        5, 20;
        10, 30;
        20, 50;
        50, 80;
        100, 100;  % 最慢但最优
    ];
    
    num_configs = size(ga_configs, 1);
    
    ga_qoe = zeros(num_configs, 1);
    ga_time = zeros(num_configs, 1);
    proposed_qoe = 0;
    proposed_time = 0;
    random_qoe = 0;
    norm_qoe = 0;
    
    % ========== Test Proposed ==========
    fprintf('测试 Proposed...\n');
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
    fprintf('测试 Baselines...\n');
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
    fprintf('\n测试 GA 不同配置...\n');
    
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
    
    % GA曲线
    plot(ga_time, ga_qoe, '-s', 'LineWidth', 2, 'MarkerSize', 10, ...
        'MarkerFaceColor', [0.3 0.6 0.9], 'Color', [0.2 0.4 0.8], ...
        'DisplayName', 'GA');
    hold on;
    
    % Proposed点
    plot(proposed_time, proposed_qoe, 'p', 'MarkerSize', 18, ...
        'MarkerFaceColor', [0.9 0.6 0.1], 'MarkerEdgeColor', [0.7 0.4 0], ...
        'LineWidth', 2, 'DisplayName', 'Proposed');
    
    % 水平参考线
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
    fprintf('关键发现\n');
    fprintf('========================================\n');
    
    % 找到GA首次达到proposed性能的点
    match_idx = find(ga_qoe <= proposed_qoe, 1);
    if ~isempty(match_idx)
        fprintf('GA 首次达到 Proposed 性能:\n');
        fprintf('  配置: Niter=%d, Np=%d\n', ga_configs(match_idx,1), ga_configs(match_idx,2));
        fprintf('  GA时间: %.2f ms\n', ga_time(match_idx));
        fprintf('  Proposed时间: %.3f ms\n', proposed_time);
        speedup = ga_time(match_idx)/proposed_time;
        fprintf('  → Proposed 快 %.1f 倍!\n', speedup);
    else
        fprintf('在测试范围内，GA未能达到Proposed性能\n');
        fprintf('最接近的GA配置: Niter=%d, Np=%d, QoE=%.4f\n', ...
            ga_configs(end,1), ga_configs(end,2), ga_qoe(end));
        speedup = ga_time(1)/proposed_time;
        fprintf('即使最快GA配置，Proposed仍快 %.1f 倍\n', speedup);
    end
    
    % 性能提升
    fprintf('\n相比 Random: Proposed 降低 %.1f%% QoE Cost\n', ...
        (random_qoe - proposed_qoe) / random_qoe * 100);
    fprintf('相比 Norm:   Proposed 降低 %.1f%% QoE Cost\n', ...
        (norm_qoe - proposed_qoe) / norm_qoe * 100);
    
    % 关键优势总结
    fprintf('\n========================================\n');
    fprintf('Proposed算法的核心优势\n');
    fprintf('========================================\n');
    fprintf('1. 确定性结果 - GA每次运行结果不同\n');
    fprintf('2. 理论可分析 - 可证明近似比\n');
    fprintf('3. 低复杂度 - O(K²L) vs O(Np·Niter·KL)\n');
    fprintf('4. 实时可行 - 满足毫秒级决策需求\n');
    
    % ========== Save ==========
    if p.Results.save
        out_dir = fullfile(proj_root, 'figures');
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        
        filename = sprintf('ga_convergence_%s', datestr(now, 'yyyymmdd_HHMMSS'));
        saveas(gcf, fullfile(out_dir, [filename '.png']));
        fprintf('\n图片已保存: %s.png\n', filename);
    end
end

function [qoe_val, sum_rate] = evaluate_assignment(cfg, ch, assign, geom, p_dbw)
    h_eff = effective_channel(cfg, ch, assign);
    [gamma, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
    
    xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
        struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
    
    prop_delay = calc_prop_delay_conv(assign, geom);
    [qoe_val, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:), prop_delay);
end

function prop_delay = calc_prop_delay_conv(assign, geom)
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
