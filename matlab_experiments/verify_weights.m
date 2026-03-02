%VERIFY_WEIGHTS 韫囶偊鈧喖鐛欑拠浣规綀闁插秵鏅遍幇鐔糕偓褍鐤勬?
%
% 閺嶇绺鹃崑鍥啎閿?
% - 娴肩姷绮洪弬瑙勭《閿涘湤orm, GA閿涘绗夐懓鍐閻劍鍩涢惃?QoE 閺夊啴鍣搁棁鈧Ч?
% - 閸欘亝婀?Proposed 閼宠姤鐗撮幑顔芥綀闁插秶浼掑ú鏄忕殶閺佹潙灏柊宥囩摜閻?
% - 閸ョ姵顒濊ぐ鎾绘付濮瑰倸褰夐崠鏍ㄦ閿涘roposed 閼宠姤娲挎總钘夋勾濠娐ゅ喕鐎电懓绨查棁鈧Ч?
%
% 鐎圭偤鐛欑拋鎹愵吀閿?
% - GA: 閸ュ搫鐣炬导妯哄 Sum-Rate閿涘牅鍞悰銊ょ炊缂佺喍绱崠鏍ㄦ煙濞夋洩绱?
% - Proposed: 閺嶈宓侀弶鍐櫢娴兼ê瀵?QoE閿涘牊鍨滄禒顒傛畱閺傝纭堕敍?
% - Exhaustive: 鐢附娼堥柌宥囨畱 QoE 娴兼ê瀵查敍鍦檙ound Truth閿?

clear; clc;

fprintf('========================================\n');
fprintf('妤犲矁鐦夐弶鍐櫢閺佸繑鍔呴幀褝绱欓弽绋跨妇娴兼ê濞嶇仦鏇犮仛閿涘　n');
fprintf('========================================\n\n');

% Setup paths
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
cd(proj_root);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');

% Configuration - 閸掓盯鈧姵婀侀崠鍝勫瀻鎼达妇娈戦崷鐑樻珯
cfg = config();
cfg.users_per_cell = 6;
cfg.ris_per_cell = 4;
cfg.k0 = 1;
cfg.num_users = 6;
cfg.num_ris = 4;

% 閸忔娊鏁敍姘崇殶閺佸瀹抽弶鐔跺▏閸忚埖婀侀崠鍝勫瀻鎼?
cfg.deadlines = [0.0005, 0.002];  % 閺囩繝寮楅弽鑲╂畱瀵ゆ儼绻滅痪锔芥将 (0.5ms, 2ms)
cfg.dmax = 0.50;                   % 閺€鐐緱鐠囶厺绠熺痪锔芥将 (distortion < 0.50)

% Test parameters
p_dbw = -20;  % 闂勫秳缍嗛崝鐔哄芳閿涘苯顤冮崝鐘冲閹?
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
            
            out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights);
            delay_sat(tr, a) = 1 - out.delay_vio_rate_all;
            semantic_sat(tr, a) = 1 - out.semantic_vio_rate_all;
            qoe_cost(tr, a) = out.avg_qoe;
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
    
    fprintf('\n閸忔娊鏁€佃鐦?\n');
    if mean_delay(idx_proposed) > mean_delay(idx_ga)
        fprintf('閴?Proposed 瀵ゆ儼绻滃陇鍐婚悳?(%.1f%%) > GA (%.1f%%)\n', ...
            mean_delay(idx_proposed)*100, mean_delay(idx_ga)*100);
    else
        fprintf('閳?Proposed 瀵ゆ儼绻滃陇鍐婚悳?(%.1f%%) vs GA (%.1f%%)\n', ...
            mean_delay(idx_proposed)*100, mean_delay(idx_ga)*100);
    end
    
    if mean_semantic(idx_proposed) > mean_semantic(idx_ga)
        fprintf('閴?Proposed 鐠囶厺绠熷陇鍐婚悳?(%.1f%%) > GA (%.1f%%)\n', ...
            mean_semantic(idx_proposed)*100, mean_semantic(idx_ga)*100);
    else
        fprintf('閳?Proposed 鐠囶厺绠熷陇鍐婚悳?(%.1f%%) vs GA (%.1f%%)\n', ...
            mean_semantic(idx_proposed)*100, mean_semantic(idx_ga)*100);
    end
end

fprintf('\n========================================\n');
fprintf('妤犲矁鐦夌€瑰本鍨歕n');
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
            % GA: 閸ュ搫鐣炬导妯哄 Sum-Rate閿涘牅鍞悰銊ょ炊缂佺喍绱崠鏍ㄦ煙濞夋洩绱濇稉宥堚偓鍐閺夊啴鍣搁敍?
            opts.verbose = false;
            opts.semantic_mode = sm;
            opts.table_path = tp;
            opts.geom = geom;
            opts.optimize_sumrate = true;  % 娴肩姷绮洪弬瑙勭《閿涙俺鎷峰Ч鍌炩偓鐔哄芳
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights(:).', cfg.num_users, 1);
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
end

