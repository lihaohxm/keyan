%VERIFY_LOGIC 验证实验设计的科学逻辑
%
% 实验设计（时间公平对比）：
% - Proposed: QoE-aware 贪婪算法（~3ms）
% - GA: 遗传算法，限制迭代次数（~10-20ms，公平对比）
% - Exhaustive: 穷举搜索（Ground Truth，但慢）
% - Norm: 基于信道增益（传统贪婪）
% - Random: 随机基线
%
% 所有算法优化同一目标：最小化 QoE Cost
%
% 预期结果：
% 1. Exhaustive 是 Ground Truth（QoE 最低）
% 2. Proposed 接近 Exhaustive（证明算法质量）
% 3. GA（有限迭代）可能不如 Proposed（时间不够收敛）
% 4. Norm/Random 较差

clear; clc;

fprintf('========================================\n');
fprintf('验证实验设计的科学逻辑\n');
fprintf('========================================\n\n');

% Setup paths
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
cd(proj_root);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');

% Configuration
cfg = config();
cfg.users_per_cell = 6;  % Small scale for exhaustive
cfg.ris_per_cell = 4;
cfg.k0 = 1;
cfg.num_users = 6;
cfg.num_ris = 4;

% Test parameters
p_dbw = -15;  % Medium power
seed = 42;
mc = 10;  % Quick test

algorithms = {'random', 'norm', 'proposed', 'exhaustive', 'ga'};
A = numel(algorithms);

% Results storage
qoe_results = zeros(mc, A);
sr_results = zeros(mc, A);
time_results = zeros(mc, A);

fprintf('运行 %d 次蒙特卡罗试验 (p = %d dBW)\n\n', mc, p_dbw);

for tr = 1:mc
    trial_seed = seed + tr;
    rng(trial_seed);
    
    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);
    
    for a = 1:A
        alg = algorithms{a};
        
        tic;
        assign = pick_assignment_verify(cfg, ch, alg, p_dbw, trial_seed, geom);
        time_results(tr, a) = toc;
        
        % Evaluate with effective_channel (same as optimization phase)
        h_eff = effective_channel(cfg, ch, assign);
        [gamma, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
        
        xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
            struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
        
        prop_delay = calc_prop_delay_verify(assign, geom, cfg);
        [avg_qoe, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:), prop_delay);
        
        qoe_results(tr, a) = avg_qoe;
        sr_results(tr, a) = sum_rate;
    end
end

% Print results
fprintf('========================================\n');
fprintf('结果汇总 (Mean ± Std)\n');
fprintf('========================================\n\n');

fprintf('%-12s | %15s | %15s | %12s\n', 'Algorithm', 'QoE Cost', 'Sum-Rate', 'Time (ms)');
fprintf('%s\n', repmat('-', 1, 60));

for a = 1:A
    fprintf('%-12s | %6.3f ± %5.3f | %6.2e ± %4.2e | %6.2f\n', ...
        algorithms{a}, ...
        mean(qoe_results(:, a)), std(qoe_results(:, a)), ...
        mean(sr_results(:, a)), std(sr_results(:, a)), ...
        mean(time_results(:, a)) * 1000);
end

fprintf('\n========================================\n');
fprintf('逻辑验证\n');
fprintf('========================================\n\n');

mean_qoe = mean(qoe_results, 1);
mean_sr = mean(sr_results, 1);

% Find indices
idx_random = 1;
idx_norm = 2;
idx_proposed = 3;
idx_exhaustive = 4;
idx_ga = 5;

% Check 1: Exhaustive should have lowest QoE (Ground Truth)
[min_qoe, min_idx] = min(mean_qoe);
if min_idx == idx_exhaustive
    fprintf('✓ Exhaustive 有最低的 QoE Cost (%.3f) - Ground Truth\n', min_qoe);
else
    fprintf('○ %s (%.3f) 低于 Exhaustive (%.3f)\n', ...
        algorithms{min_idx}, min_qoe, mean_qoe(idx_exhaustive));
end

% Check 2: Proposed should be close to Exhaustive
gap_proposed = (mean_qoe(idx_proposed) - mean_qoe(idx_exhaustive)) / mean_qoe(idx_exhaustive) * 100;
fprintf('  Proposed 与 Exhaustive 差距: %.1f%%\n', gap_proposed);
if gap_proposed < 5
    fprintf('✓ Proposed 非常接近最优！\n');
elseif gap_proposed < 15
    fprintf('✓ Proposed 接近最优（可接受）\n');
else
    fprintf('○ Proposed 与最优差距较大\n');
end

% Check 3: Proposed vs GA (time-fair comparison)
if mean_qoe(idx_proposed) < mean_qoe(idx_ga)
    improvement = (mean_qoe(idx_ga) - mean_qoe(idx_proposed)) / mean_qoe(idx_ga) * 100;
    fprintf('✓ Proposed (%.3f) 比 GA (%.3f) 好 %.1f%% - 时间公平对比下获胜！\n', ...
        mean_qoe(idx_proposed), mean_qoe(idx_ga), improvement);
elseif abs(mean_qoe(idx_proposed) - mean_qoe(idx_ga)) / mean_qoe(idx_ga) < 0.05
    fprintf('○ Proposed (%.3f) 与 GA (%.3f) 相当\n', ...
        mean_qoe(idx_proposed), mean_qoe(idx_ga));
else
    fprintf('○ GA (%.3f) 略优于 Proposed (%.3f)\n', ...
        mean_qoe(idx_ga), mean_qoe(idx_proposed));
end

% Check 4: Proposed vs Norm
if mean_qoe(idx_proposed) < mean_qoe(idx_norm)
    improvement = (mean_qoe(idx_norm) - mean_qoe(idx_proposed)) / mean_qoe(idx_norm) * 100;
    fprintf('✓ Proposed (%.3f) 比 Norm (%.3f) 好 %.1f%%\n', ...
        mean_qoe(idx_proposed), mean_qoe(idx_norm), improvement);
else
    fprintf('○ Proposed 与 Norm 差距不明显\n');
end

% Check 5: Time comparison
mean_time = mean(time_results, 1) * 1000;  % ms
fprintf('\n运行时间对比:\n');
fprintf('  Proposed: %.1f ms\n', mean_time(idx_proposed));
fprintf('  GA:       %.1f ms\n', mean_time(idx_ga));
fprintf('  Exhaustive: %.1f ms\n', mean_time(idx_exhaustive));
if mean_time(idx_proposed) < mean_time(idx_ga)
    fprintf('✓ Proposed 比 GA 快 %.1fx\n', mean_time(idx_ga)/mean_time(idx_proposed));
end

% Check 6: Random should be worst
if mean_qoe(idx_random) == max(mean_qoe)
    fprintf('✓ Random 的 QoE Cost 最高 (%.3f) - 基线正确\n', mean_qoe(idx_random));
end

fprintf('\n========================================\n');
fprintf('验证完成\n');
fprintf('========================================\n');

%% Helper functions
function assign = pick_assignment_verify(cfg, ch, alg, p_dbw, seed, geom)
    if isfield(cfg,'weights') && ~isempty(cfg.weights)
        w = cfg.weights(1,:);
    else
        w = [0.5 0.5];
    end
    sm = cfg.semantic_mode;
    tp = cfg.semantic_table;

    switch alg
        case 'random'
            assign = random_match(cfg, seed);
        case 'norm'
            assign = norm_based(cfg, ch);
        case 'proposed'
            % OUR METHOD: optimize QoE Cost
            assign = qoe_aware(cfg, ch, p_dbw, sm, tp, w, geom);
        case 'exhaustive'
            % Ground Truth: minimize QoE Cost
            ex_opts.optimize_qoe = true;
            ex_opts.geom = geom;
            [assign, info] = exhaustive(cfg, ch, p_dbw, ex_opts);
            if info.skipped
                assign = norm_based(cfg, ch);
            end
        case 'ga'
            % GA with LIMITED iterations (fair time comparison)
            opts.verbose = false;
            opts.semantic_mode = sm;
            opts.table_path = tp;
            opts.weights = w;
            opts.geom = geom;
            opts.optimize_sumrate = false;  % Optimize QoE
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

function prop_delay = calc_prop_delay_verify(assign, geom, cfg)
    num_users = numel(assign);
    prop_delay = zeros(num_users, 1);
    prop_delay_factor = 1e-5;  % Must match qoe_aware.m
    
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
            total_dist = d_bs_ris + d_ris_ue;
            prop_delay(k) = prop_delay_factor * total_dist;
        end
    end
end
