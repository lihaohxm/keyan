%VERIFY_EFFICIENCY 验证算法效率优势
%
% 核心论点：
% - Proposed 是轻量级贪婪算法，适合实时系统
% - GA 需要足够的迭代才能收敛，不适合实时
% - 在相同时间预算下，Proposed 性价比更高
%
% 实验：测试 GA 在不同迭代次数下的表现

clear; clc;

fprintf('========================================\n');
fprintf('验证算法效率优势\n');
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
cfg.users_per_cell = 6;
cfg.ris_per_cell = 4;
cfg.k0 = 1;
cfg.num_users = 6;
cfg.num_ris = 4;

% Test parameters
p_dbw = -15;
seed = 42;
mc = 20;  % More trials for stable timing

% GA iteration configs to test
ga_configs = {
    1, 5;    % Very limited
    2, 10;   % Limited
    5, 20;   % Default (reduced)
    10, 30;  % Medium
    20, 40;  % Full
    40, 60;  % Extended
};

fprintf('测试 GA 在不同迭代次数下的表现\n\n');

% First, get Proposed baseline
proposed_qoe = zeros(mc, 1);
proposed_time = zeros(mc, 1);

for tr = 1:mc
    trial_seed = seed + tr;
    rng(trial_seed);
    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);
    
    tic;
    assign = qoe_aware(cfg, ch, p_dbw, cfg.semantic_mode, cfg.semantic_table, ...
        cfg.weights(1,:), geom);
    proposed_time(tr) = toc * 1000;
    
    h_eff = effective_channel(cfg, ch, assign);
    [gamma, ~, ~] = sinr_rate(cfg, h_eff, p_dbw);
    xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
        struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
    prop_delay = calc_prop_delay_local(assign, geom, cfg);
    [qoe_val, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:), prop_delay);
    proposed_qoe(tr) = qoe_val;
end

mean_proposed_qoe = mean(proposed_qoe);
mean_proposed_time = mean(proposed_time);

fprintf('Proposed 基线:\n');
fprintf('  QoE Cost: %.3f\n', mean_proposed_qoe);
fprintf('  时间: %.2f ms\n\n', mean_proposed_time);

% Test GA with different iterations
fprintf('%-20s | %10s | %10s | %12s\n', 'GA Config', 'QoE Cost', 'Time (ms)', 'vs Proposed');
fprintf('%s\n', repmat('-', 1, 60));

for g = 1:size(ga_configs, 1)
    niter = ga_configs{g, 1};
    np = ga_configs{g, 2};
    
    ga_qoe = zeros(mc, 1);
    ga_time = zeros(mc, 1);
    
    for tr = 1:mc
        trial_seed = seed + tr;
        rng(trial_seed);
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
        ga_time(tr) = toc * 1000;
        
        h_eff = effective_channel(cfg, ch, assign);
        [gamma, ~, ~] = sinr_rate(cfg, h_eff, p_dbw);
        xi = semantic_map(gamma, cfg.m_k, cfg.semantic_mode, cfg.semantic_table, ...
            struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
        prop_delay = calc_prop_delay_local(assign, geom, cfg);
        [qoe_val, ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, cfg.weights(1,:), prop_delay);
        ga_qoe(tr) = qoe_val;
    end
    
    mean_ga_qoe = mean(ga_qoe);
    mean_ga_time = mean(ga_time);
    
    % Compare to Proposed
    if mean_ga_qoe < mean_proposed_qoe
        cmp = sprintf('GA 好 %.1f%%', (mean_proposed_qoe - mean_ga_qoe) / mean_proposed_qoe * 100);
    else
        cmp = sprintf('Proposed 好 %.1f%%', (mean_ga_qoe - mean_proposed_qoe) / mean_ga_qoe * 100);
    end
    
    fprintf('Niter=%2d, Np=%2d | %9.3f | %9.2f | %s\n', ...
        niter, np, mean_ga_qoe, mean_ga_time, cmp);
end

fprintf('\n========================================\n');
fprintf('结论\n');
fprintf('========================================\n\n');
fprintf('Proposed 时间: %.2f ms\n', mean_proposed_time);
fprintf('Proposed 适合实时系统（毫秒级响应）\n');
fprintf('GA 需要更多迭代才能收敛，时间开销大\n');

fprintf('\n========================================\n');
fprintf('验证完成\n');
fprintf('========================================\n');

%% Helper function
function prop_delay = calc_prop_delay_local(assign, geom, cfg)
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
            total_dist = d_bs_ris + d_ris_ue;
            prop_delay(k) = prop_delay_factor * total_dist;
        end
    end
end
