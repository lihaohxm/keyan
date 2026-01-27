function run_id = exp_urgent(varargin)
%EXP_URGENT 紧迫用户场景实验
%
% 场景设计：
%   - 用户1-4：边缘用户（远离BS，NLoS，直连极差）
%   - 用户5-12：普通用户（靠近BS，LoS，直连正常）
%   - RIS：部署在cell-edge方向，能覆盖边缘用户
%
% 预期结果：
%   - Proposed能识别边缘用户并优先分配RIS
%   - Random可能把RIS浪费在普通用户上
%   - Proposed的QoE优势会更明显

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
    
    fprintf('========================================\n');
    fprintf('紧迫用户场景实验\n');
    fprintf('========================================\n\n');
    
    % 基础配置
    cfg = config();
    cfg.users_per_cell = 12;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 12;
    cfg.num_ris = 4;
    
    % 紧迫场景专用参数
    cfg.n_ris = 64;        % RIS元素数量: 36
    cfg.dmax = 0.30;       % 语义失真阈值: D<=0.30, 即 xi>=0.70
    
    % 紧迫用户数量
    num_urgent = 4;  % 用户1-4是紧迫用户
    
    fprintf('配置:\n');
    fprintf('  总用户数 K = %d\n', cfg.num_users);
    fprintf('  紧迫用户 = %d (边缘, NLoS)\n', num_urgent);
    fprintf('  普通用户 = %d (近距, LoS)\n', cfg.num_users - num_urgent);
    fprintf('  RIS数量 L = %d\n', cfg.num_ris);
    fprintf('  RIS元素数 N = %d\n', cfg.n_ris);
    fprintf('  每RIS容量 k0 = %d\n', cfg.k0);
    fprintf('  语义门槛 xi_th = %.2f (dmax=%.2f)\n', 1-cfg.dmax, cfg.dmax);
    fprintf('  MC次数 = %d\n\n', mc);
    
    algorithms = {'random', 'norm', 'proposed', 'ga'};
    p_dbw_list = linspace(-25, -5, 8);
    
    A = numel(algorithms);
    X = numel(p_dbw_list);
    
    % 结果存储
    q_all = zeros(mc, X, A);
    sr_all = zeros(mc, X, A);
    urgent_qoe_all = zeros(mc, X, A);
    normal_qoe_all = zeros(mc, X, A);
    violation_all = zeros(mc, X, A);  % 时延违约率
    semantic_vio_all = zeros(mc, X, A);  % 语义违约率
    
    % 诊断数据：语义相似度和SINR
    xi_urgent_all = zeros(mc, X, A);  % 边缘用户平均xi
    xi_normal_all = zeros(mc, X, A);  % 普通用户平均xi
    sinr_urgent_all = zeros(mc, X, A);  % 边缘用户平均SINR(dB)
    sinr_normal_all = zeros(mc, X, A);  % 普通用户平均SINR(dB)
    
    for tr = 1:mc
        trial_seed = base_seed + tr;
        
        % 生成紧迫场景的几何和信道
        [geom, ch] = generate_urgent_scenario(cfg, trial_seed, num_urgent);
        
        for ix = 1:X
            p_dbw = p_dbw_list(ix);
            
            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);
                
                h_eff = effective_channel(cfg, ch, assign);
                [gamma, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
                
                xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
                    struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
                
                prop_delay = calc_prop_delay(assign, geom);
                [avg_qoe, qoe_vec, meta] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:), prop_delay);
                
                q_all(tr, ix, a) = avg_qoe;
                sr_all(tr, ix, a) = sum_rate;
                urgent_qoe_all(tr, ix, a) = mean(qoe_vec(1:num_urgent));
                normal_qoe_all(tr, ix, a) = mean(qoe_vec(num_urgent+1:end));
                violation_all(tr, ix, a) = meta.delay_violation_rate;
                semantic_vio_all(tr, ix, a) = meta.semantic_violation_rate;
                
                % 诊断数据
                xi_urgent_all(tr, ix, a) = mean(xi(1:num_urgent));
                xi_normal_all(tr, ix, a) = mean(xi(num_urgent+1:end));
                sinr_db = 10 * log10(gamma + 1e-12);
                sinr_urgent_all(tr, ix, a) = mean(sinr_db(1:num_urgent));
                sinr_normal_all(tr, ix, a) = mean(sinr_db(num_urgent+1:end));
            end
        end
        
        if mod(tr, 50) == 0
            fprintf('  MC trial %d/%d\n', tr, mc);
        end
    end
    
    % 汇总结果
    result = struct();
    result.x_vals = p_dbw_list;
    result.algorithms = algorithms;
    result.sum_rate = squeeze(mean(sr_all, 1));
    result.avg_qoe = squeeze(mean(q_all, 1));
    result.urgent_qoe = squeeze(mean(urgent_qoe_all, 1));
    result.normal_qoe = squeeze(mean(normal_qoe_all, 1));
    result.violation_rate = squeeze(mean(violation_all, 1));
    result.semantic_violation_rate = squeeze(mean(semantic_vio_all, 1));
    result.xi_urgent = squeeze(mean(xi_urgent_all, 1));
    result.xi_normal = squeeze(mean(xi_normal_all, 1));
    result.sinr_urgent = squeeze(mean(sinr_urgent_all, 1));
    result.sinr_normal = squeeze(mean(sinr_normal_all, 1));
    result.experiment = 'urgent';
    
    % 打印关键结果
    fprintf('\n========================================\n');
    fprintf('关键结果 (P = -15 dBW)\n');
    fprintf('========================================\n');
    mid_idx = 4;  % -15 dBW 附近
    fprintf('%-12s %12s %12s %12s %12s\n', '算法', 'Avg QoE', '紧迫用户QoE', '普通用户QoE', '违约率');
    fprintf('%s\n', repmat('-', 1, 60));
    for a = 1:A
        fprintf('%-12s %12.3f %12.3f %12.3f %12.1f%%\n', ...
            algorithms{a}, ...
            result.avg_qoe(mid_idx, a), ...
            result.urgent_qoe(mid_idx, a), ...
            result.normal_qoe(mid_idx, a), ...
            result.violation_rate(mid_idx, a) * 100);
    end
    
    % 诊断信息：语义相似度和SINR
    fprintf('\n========================================\n');
    fprintf('诊断信息: 语义相似度 xi 和 SINR\n');
    fprintf('========================================\n');
    fprintf('语义约束: D <= %.2f, 即 xi >= %.2f\n', cfg.dmax, 1 - cfg.dmax);
    fprintf('RIS元素数量 N = %d\n', cfg.n_ris);
    fprintf('DeepSC表(M=8): SNR>=4dB才能达到xi>=0.70\n\n');
    
    fprintf('%-12s %10s %10s %10s %10s\n', '算法', 'xi边缘', 'xi普通', 'SNR边缘', 'SNR普通');
    fprintf('%s\n', repmat('-', 1, 54));
    for a = 1:A
        fprintf('%-12s %10.3f %10.3f %10.1f dB %10.1f dB\n', ...
            algorithms{a}, ...
            result.xi_urgent(mid_idx, a), ...
            result.xi_normal(mid_idx, a), ...
            result.sinr_urgent(mid_idx, a), ...
            result.sinr_normal(mid_idx, a));
    end
    
    fprintf('\n结论: ');
    if result.xi_urgent(mid_idx, 3) < (1 - cfg.dmax) && result.xi_normal(mid_idx, 3) < (1 - cfg.dmax)
        fprintf('当前功率范围内，所有用户的xi均低于%.2f，语义约束不可行。\n', 1 - cfg.dmax);
    elseif result.xi_urgent(mid_idx, 3) < (1 - cfg.dmax)
        fprintf('仅边缘用户xi低于%.2f，语义约束对边缘用户不可行。\n', 1 - cfg.dmax);
    else
        fprintf('语义约束可行。\n');
    end
    
    % 保存和绘图
    run_id = sprintf('urgent_%s', datestr(now, 'yyyymmdd_HHMMSS'));
    
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    plot_urgent_results(run_id, result, out_dir, algorithms);
    
    fprintf('\n实验完成！run_id: %s\n', run_id);
    cd(old_dir);
end

%% 生成紧迫场景
function [geom, ch] = generate_urgent_scenario(cfg, seed, num_urgent)
    rng(seed, 'twister');
    
    num_users = cfg.num_users;
    num_ris = cfg.num_ris;
    
    bs_pos = [0, 0];
    ue = zeros(num_users, 2);
    ris = zeros(num_ris, 2);
    
    % ========== 用户位置 ==========
    % 紧迫用户(1-num_urgent): 边缘位置，120-180m，集中在一个扇区
    urgent_sector = pi/4 + 0.3*randn();  % 紧迫用户所在扇区
    for k = 1:num_urgent
        r = 120 + 60 * rand();  % 120-180m，远离BS
        ang = urgent_sector + 0.3 * (rand() - 0.5);  % 集中在一个扇区
        ue(k, :) = bs_pos + r * [cos(ang), sin(ang)];
    end
    
    % 普通用户(num_urgent+1:end): 近距离，30-70m，均匀分布
    for k = num_urgent+1:num_users
        r = 30 + 40 * rand();  % 30-70m，靠近BS
        ang = 2 * pi * rand();  % 均匀分布
        ue(k, :) = bs_pos + r * [cos(ang), sin(ang)];
    end
    
    % ========== RIS位置 ==========
    % RIS部署在紧迫用户方向，80-100m，能覆盖边缘用户
    for l = 1:num_ris
        r_dist = 80 + 20 * rand();  % 80-100m
        r_ang = urgent_sector + 0.5 * (l - 1 - num_ris/2) / num_ris * pi;  % 扇形分布
        ris(l, :) = bs_pos + r_dist * [cos(r_ang), sin(r_ang)];
    end
    
    geom.bs = bs_pos;
    geom.ue = ue;
    geom.ris = ris;
    
    % ========== 信道生成 ==========
    h_d = zeros(cfg.nt, num_users);
    G = zeros(cfg.nt, cfg.n_ris, num_ris);
    H_ris = zeros(cfg.n_ris, num_users, num_ris);
    
    % 路损指数
    alpha_urgent = 4.0;   % 紧迫用户：NLoS，路损大
    alpha_normal = 2.8;   % 普通用户：LoS，路损小
    alpha_br = 2.2;       % BS->RIS: LoS
    alpha_ru = 2.5;       % RIS->UE
    
    % 直连信道 BS->UE
    for k = 1:num_users
        d = norm(ue(k,:) - bs_pos) + cfg.eps;
        
        if k <= num_urgent
            % 紧迫用户：NLoS，大路损
            alpha = alpha_urgent;
        else
            % 普通用户：LoS，小路损
            alpha = alpha_normal;
        end
        
        pathloss = d.^(-alpha);
        h_d(:, k) = sqrt(pathloss/2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
    end
    
    % 级联信道 BS->RIS->UE
    for l = 1:num_ris
        d_br = norm(ris(l,:) - bs_pos) + cfg.eps;
        pl_br = d_br.^(-alpha_br);
        G(:, :, l) = sqrt(pl_br/2) * (randn(cfg.nt, cfg.n_ris) + 1j * randn(cfg.nt, cfg.n_ris));
        
        for k = 1:num_users
            d_ru = norm(ue(k,:) - ris(l,:)) + cfg.eps;
            pl_ru = d_ru.^(-alpha_ru);
            H_ris(:, k, l) = sqrt(pl_ru/2) * (randn(cfg.n_ris, 1) + 1j * randn(cfg.n_ris, 1));
        end
    end
    
    % RIS相位（对齐模式）
    theta = exp(1j * 2 * pi * rand(cfg.n_ris, num_ris));
    
    ch.h_d = h_d;
    ch.G = G;
    ch.H_ris = H_ris;
    ch.theta = theta;
end

%% 算法选择
function assign = pick_assignment(cfg, ch, alg, p_dbw, seed, geom)
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

%% 传播时延计算
function prop_delay = calc_prop_delay(assign, geom)
    num_users = numel(assign);
    prop_delay = zeros(num_users, 1);
    prop_delay_factor = 1e-7;
    
    if isempty(geom) || ~isfield(geom, 'ris') || ~isfield(geom, 'ue')
        return;
    end
    
    bs_pos = geom.bs;
    if numel(bs_pos) > 2, bs_pos = bs_pos(1:2); end
    
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

%% 绘图函数
function plot_urgent_results(run_id, result, out_dir, algorithms)
    x = result.x_vals;
    A = numel(algorithms);
    markers = {'o', 's', 'd', '^'};
    colors = lines(A);
    
    % ========== 图1: 平均QoE ==========
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.avg_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', algorithms{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Avg QoE Cost');
    title('紧迫场景: 全体用户平均QoE');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_avg_qoe.png', run_id)));
    close(gcf);
    
    % ========== 图2: 紧迫用户QoE ==========
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.urgent_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', algorithms{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Urgent Users QoE Cost');
    title('紧迫场景: 边缘用户QoE (1-4, NLoS)');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_urgent_qoe.png', run_id)));
    close(gcf);
    
    % ========== 图3: 普通用户QoE ==========
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.normal_qoe(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', algorithms{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Normal Users QoE Cost');
    title('紧迫场景: 普通用户QoE (5-12, LoS)');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_normal_qoe.png', run_id)));
    close(gcf);
    
    % ========== 图4: Sum-rate ==========
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.sum_rate(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', algorithms{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Sum-rate (bps)');
    title('紧迫场景: Sum-rate');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_sum_rate.png', run_id)));
    close(gcf);
    
    % ========== 图5: 时延违约率 ==========
    figure('Position', [100 100 700 500]);
    hold on;
    for a = 1:A
        plot(x, result.violation_rate(:, a) * 100, ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', algorithms{a});
    end
    hold off;
    xlabel('p (dBW)');
    ylabel('Delay Violation Rate (%)');
    title('紧迫场景: 时延违约率');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_delay_violation.png', run_id)));
    close(gcf);
    
    % ========== 图6: 语义相似度 xi (替代语义违约率) ==========
    figure('Position', [100 100 700 500]);
    hold on;
    % 绘制边缘用户xi（实线）
    for a = 1:A
        plot(x, result.xi_urgent(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', [algorithms{a} ' (边缘)']);
    end
    % 绘制普通用户xi（虚线）
    for a = 1:A
        plot(x, result.xi_normal(:, a), ['--' markers{a}], ...
            'LineWidth', 1.2, 'MarkerSize', 6, 'Color', colors(a,:), ...
            'DisplayName', [algorithms{a} ' (普通)']);
    end
    % 语义约束线 (xi = 0.7)
    yline(0.7, 'r--', 'LineWidth', 2, 'DisplayName', '\xi_{th}=0.7');
    hold off;
    xlabel('p (dBW)');
    ylabel('Semantic Similarity \xi');
    title('紧迫场景: 语义相似度 (实线=边缘, 虚线=普通)');
    legend('Location', 'eastoutside', 'FontSize', 8);
    grid on;
    ylim([0 1]);
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_semantic_xi.png', run_id)));
    close(gcf);
    
    % ========== 图7: SINR分布 ==========
    figure('Position', [100 100 700 500]);
    hold on;
    % 绘制边缘用户SINR（实线）
    for a = 1:A
        plot(x, result.sinr_urgent(:, a), ['-' markers{a}], ...
            'LineWidth', 1.5, 'MarkerSize', 8, 'Color', colors(a,:), ...
            'DisplayName', [algorithms{a} ' (边缘)']);
    end
    % 绘制普通用户SINR（虚线）
    for a = 1:A
        plot(x, result.sinr_normal(:, a), ['--' markers{a}], ...
            'LineWidth', 1.2, 'MarkerSize', 6, 'Color', colors(a,:), ...
            'DisplayName', [algorithms{a} ' (普通)']);
    end
    % 参考线: SNR=4dB (xi=0.7门槛)
    yline(4, 'r--', 'LineWidth', 2, 'DisplayName', 'SNR_{th}=4dB');
    hold off;
    xlabel('p (dBW)');
    ylabel('Average SINR (dB)');
    title('紧迫场景: SINR分布 (实线=边缘, 虚线=普通)');
    legend('Location', 'eastoutside', 'FontSize', 8);
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_sinr.png', run_id)));
    close(gcf);
    
    fprintf('图片已保存:\n');
    fprintf('  urgent_%s_avg_qoe.png\n', run_id);
    fprintf('  urgent_%s_urgent_qoe.png\n', run_id);
    fprintf('  urgent_%s_normal_qoe.png\n', run_id);
    fprintf('  urgent_%s_sum_rate.png\n', run_id);
    fprintf('  urgent_%s_delay_violation.png\n', run_id);
    fprintf('  urgent_%s_semantic_xi.png\n', run_id);
    fprintf('  urgent_%s_sinr.png\n', run_id);
end
