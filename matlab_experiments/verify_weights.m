%VERIFY_WEIGHTS 快速验证权重敏感性实验
%
% 核心假设：
% - 传统方法（Norm, GA）不考虑用户的 QoE 权重需求
% - 只有 Proposed 能根据权重灵活调整匹配策略
% - 因此当需求变化时，Proposed 能更好地满足对应需求
%
% 实验设计：
% - GA: 固定优化 Sum-Rate（代表传统优化方法）
% - Proposed: 根据权重优化 QoE（我们的方法）
% - Exhaustive: 带权重的 QoE 优化（Ground Truth）

clear; clc;

fprintf('========================================\n');
fprintf('验证权重敏感性（核心优势展示）\n');
fprintf('========================================\n\n');

% Setup paths
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
cd(proj_root);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');

% Configuration - 创造有区分度的场景
cfg = config();
cfg.users_per_cell = 6;
cfg.ris_per_cell = 4;
cfg.k0 = 1;
cfg.num_users = 6;
cfg.num_ris = 4;

% 关键：调整约束使其有区分度
cfg.deadlines = [0.0005, 0.002];  % 更严格的延迟约束 (0.5ms, 2ms)
cfg.dmax = 0.50;                   % 放松语义约束 (distortion < 0.50)

% Test parameters
p_dbw = -20;  % 降低功率，增加挑战
seed = 42;
mc = 10;  % Quick test

% Test 3 weight configs: delay-focused, balanced, semantic-focused
weight_configs = {
    [0.9, 0.1], 'Delay-focused (w_d=0.9)';
    [0.5, 0.5], 'Balanced (w_d=0.5)';
    [0.1, 0.9], 'Semantic-focused (w_d=0.1)';
};

algorithms = {'random', 'norm', 'proposed', 'exhaustive', 'ga'};
A = numel(algorithms);

for w_idx = 1:size(weight_configs, 1)
    weights = weight_configs{w_idx, 1};
    config_name = weight_configs{w_idx, 2};
    
    fprintf('\n========================================\n');
    fprintf('%s\n', config_name);
    fprintf('========================================\n');
    
    delay_sat = zeros(mc, A);
    semantic_sat = zeros(mc, A);
    qoe_cost = zeros(mc, A);
    time_ms = zeros(mc, A);
    
    for tr = 1:mc
        trial_seed = seed + tr;
        rng(trial_seed);
        
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        
        for a = 1:A
            alg = algorithms{a};
            
            tic;
            assign = pick_assign(cfg, ch, alg, p_dbw, trial_seed, geom, weights);
            time_ms(tr, a) = toc * 1000;
            
            h_eff = effective_channel(cfg, ch, assign);
            [gamma, ~, ~] = sinr_rate(cfg, h_eff, p_dbw);
            
            xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
                struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
            
            prop_delay = calc_prop_delay_local(assign, geom, cfg);
            [qoe_val, ~, meta] = qoe(cfg, gamma, cfg.m_k, xi, weights, prop_delay);
            
            delay_sat(tr, a) = meta.deadline_ok;
            semantic_sat(tr, a) = meta.semantic_ok;
            qoe_cost(tr, a) = qoe_val;
        end
    end
    
    % Print results
    fprintf('\n%-12s | %10s | %12s | %10s | %8s\n', ...
        'Algorithm', 'Delay Sat', 'Semantic Sat', 'QoE Cost', 'Time(ms)');
    fprintf('%s\n', repmat('-', 1, 65));
    
    for a = 1:A
        fprintf('%-12s | %9.1f%% | %11.1f%% | %9.3f | %7.1f\n', ...
            algorithms{a}, ...
            mean(delay_sat(:, a)) * 100, ...
            mean(semantic_sat(:, a)) * 100, ...
            mean(qoe_cost(:, a)), ...
            mean(time_ms(:, a)));
    end
    
    % Highlight key comparisons
    mean_delay = mean(delay_sat, 1);
    mean_semantic = mean(semantic_sat, 1);
    mean_qoe = mean(qoe_cost, 1);
    
    idx_proposed = 3;
    idx_ga = 5;
    idx_norm = 2;
    
    fprintf('\n关键对比:\n');
    if mean_delay(idx_proposed) > mean_delay(idx_ga)
        fprintf('✓ Proposed 延迟满足率 (%.1f%%) > GA (%.1f%%)\n', ...
            mean_delay(idx_proposed)*100, mean_delay(idx_ga)*100);
    else
        fprintf('○ Proposed 延迟满足率 (%.1f%%) vs GA (%.1f%%)\n', ...
            mean_delay(idx_proposed)*100, mean_delay(idx_ga)*100);
    end
    
    if mean_semantic(idx_proposed) > mean_semantic(idx_ga)
        fprintf('✓ Proposed 语义满足率 (%.1f%%) > GA (%.1f%%)\n', ...
            mean_semantic(idx_proposed)*100, mean_semantic(idx_ga)*100);
    else
        fprintf('○ Proposed 语义满足率 (%.1f%%) vs GA (%.1f%%)\n', ...
            mean_semantic(idx_proposed)*100, mean_semantic(idx_ga)*100);
    end
end

fprintf('\n========================================\n');
fprintf('验证完成\n');
fprintf('========================================\n');

%% Helper functions
function assign = pick_assign(cfg, ch, alg, p_dbw, seed, geom, weights)
    sm = cfg.semantic_mode;
    tp = cfg.semantic_table;

    switch alg
        case 'random'
            assign = random_match(cfg, seed);
        case 'norm'
            assign = norm_based(cfg, ch);
        case 'proposed'
            assign = qoe_aware(cfg, ch, p_dbw, sm, tp, weights, geom);
        case 'exhaustive'
            ex_opts.optimize_qoe = true;
            ex_opts.geom = geom;
            ex_opts.weights = weights;
            [assign, info] = exhaustive(cfg, ch, p_dbw, ex_opts);
            if info.skipped
                assign = norm_based(cfg, ch);
            end
        case 'ga'
            % GA: 固定优化 Sum-Rate（代表传统优化方法，不考虑权重）
            opts.verbose = false;
            opts.semantic_mode = sm;
            opts.table_path = tp;
            opts.geom = geom;
            opts.optimize_sumrate = true;  % 传统方法：追求速率
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

function prop_delay = calc_prop_delay_local(assign, geom, cfg)
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
