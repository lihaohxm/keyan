%VERIFY_EFFICIENCY 妤犲矁鐦夌粻妤佺《閺佸牏宸兼导妯哄◢
%
% 閺嶇绺剧拋铏瑰仯閿?
% - Proposed 閺勵垵浜ら柌蹇曢獓鐠愵亜鈹嗙粻妤佺《閿涘矂鈧倸鎮庣€圭偞妞傜化鑽ょ埠
% - GA 闂団偓鐟曚浇鍐绘径鐔烘畱鏉╊厺鍞幍宥堝厴閺€鑸垫殐閿涘奔绗夐柅鍌氭値鐎圭偞妞?
% - 閸︺劎娴夐崥灞炬闂傛挳顣╃粻妞剧瑓閿涘roposed 閹傜幆濮ｆ梹娲挎?
%
% 鐎圭偤鐛欓敍姘ゴ鐠?GA 閸︺劋绗夐崥宀冨嚡娴狅絾顐奸弫棰佺瑓閻ㄥ嫯銆冮悳?

clear; clc;

fprintf('========================================\n');
fprintf('妤犲矁鐦夌粻妤佺《閺佸牏宸兼导妯哄◢\n');
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

fprintf('濞村鐦?GA 閸︺劋绗夐崥宀冨嚡娴狅絾顐奸弫棰佺瑓閻ㄥ嫯銆冮悳鐧╪\n');

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
    
    out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
    proposed_qoe(tr) = out.avg_qoe;
end

mean_proposed_qoe = mean(proposed_qoe);
mean_proposed_time = mean(proposed_time);

fprintf('Proposed 閸╄櫣鍤?\n');
fprintf('  QoE Cost: %.3f\n', mean_proposed_qoe);
fprintf('  閺冨爼妫? %.2f ms\n\n', mean_proposed_time);

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
        
        out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
        ga_qoe(tr) = out.avg_qoe;
    end
    
    mean_ga_qoe = mean(ga_qoe);
    mean_ga_time = mean(ga_time);
    
    % Compare to Proposed
    if mean_ga_qoe < mean_proposed_qoe
        cmp = sprintf('GA 婵?%.1f%%', (mean_proposed_qoe - mean_ga_qoe) / mean_proposed_qoe * 100);
    else
        cmp = sprintf('Proposed 婵?%.1f%%', (mean_ga_qoe - mean_proposed_qoe) / mean_ga_qoe * 100);
    end
    
    fprintf('Niter=%2d, Np=%2d | %9.3f | %9.2f | %s\n', ...
        niter, np, mean_ga_qoe, mean_ga_time, cmp);
end

fprintf('\n========================================\n');
fprintf('缂佹捁顔慭n');
fprintf('========================================\n\n');
fprintf('Proposed 閺冨爼妫? %.2f ms\n', mean_proposed_time);
fprintf('Proposed 闁倸鎮庣€圭偞妞傜化鑽ょ埠閿涘牊顕犵粔鎺旈獓閸濆秴绨查敍濉');
fprintf('GA 闂団偓鐟曚焦娲挎径姘冲嚡娴狅絾澧犻懗鑺ユ暪閺佹冻绱濋弮鍫曟？瀵偓闁库偓婢额湤n');

fprintf('\n========================================\n');
fprintf('妤犲矁鐦夌€瑰本鍨歕n');
fprintf('========================================\n');

%% Helper function
function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights(:).', cfg.num_users, 1);
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
end

