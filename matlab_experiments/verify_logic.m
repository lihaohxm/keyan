%VERIFY_LOGIC 妤犲矁鐦夌€圭偤鐛欑拋鎹愵吀閻ㄥ嫮顫栫€涳箓鈧槒绶?
%
% 鐎圭偤鐛欑拋鎹愵吀閿涘牊妞傞梻鏉戝彆楠炲啿顕В鏃撶礆閿?
% - Proposed: QoE-aware 鐠愵亜鈹嗙粻妤佺《閿涘瀫3ms閿?
% - GA: 闁ぞ绱剁粻妤佺《閿涘矂妾洪崚鎯板嚡娴狅絾顐奸弫甯礄~10-20ms閿涘苯鍙曢獮鍐差嚠濮ｆ棑绱?
% - Exhaustive: 缁岃渹濡囬幖婊呭偍閿涘湙round Truth閿涘奔绲鹃幈顫礆
% - Norm: 閸╄桨绨穱锟犱壕婢х偟娉敍鍫滅炊缂佺喕妞芥繝顏庣礆
% - Random: 闂呭繑婧€閸╄櫣鍤?
%
% 閹碘偓閺堝鐣诲▔鏇氱喘閸栨牕鎮撴稉鈧惄顔界垼閿涙碍娓剁亸蹇撳 QoE Cost
%
% 妫板嫭婀＄紒鎾寸亯閿?
% 1. Exhaustive 閺?Ground Truth閿涘湨oE 閺堚偓娴ｅ函绱?
% 2. Proposed 閹恒儴绻?Exhaustive閿涘牐鐦夐弰搴ｇ暬濞夋洝宸濋柌蹇ョ礆
% 3. GA閿涘牊婀侀梽鎰嚡娴狅綇绱氶崣顖濆厴娑撳秴顩?Proposed閿涘牊妞傞梻缈犵瑝婢剁喐鏁归弫娑崇礆
% 4. Norm/Random 鏉堝啫妯?

clear; clc;

fprintf('========================================\n');
fprintf('妤犲矁鐦夌€圭偤鐛欑拋鎹愵吀閻ㄥ嫮顫栫€涳箓鈧槒绶玕n');
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

fprintf('鏉╂劘顢?%d 濞喡ゆ寢閻楃懓宕辩純妤勭槸妤?(p = %d dBW)\n\n', mc, p_dbw);

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
        
        out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
        qoe_results(tr, a) = out.avg_qoe;
        sr_results(tr, a) = out.sum_rate_bps;
    end
end

% Print results
fprintf('========================================\n');
fprintf('缂佹挻鐏夊Ч鍥ㄢ偓?(Mean 鍗?Std)\n');
fprintf('========================================\n\n');

fprintf('%-12s | %15s | %15s | %12s\n', 'Algorithm', 'QoE Cost', 'Sum-Rate', 'Time (ms)');
fprintf('%s\n', repmat('-', 1, 60));

for a = 1:A
    fprintf('%-12s | %6.3f 鍗?%5.3f | %6.2e 鍗?%4.2e | %6.2f\n', ...
        algorithms{a}, ...
        mean(qoe_results(:, a)), std(qoe_results(:, a)), ...
        mean(sr_results(:, a)), std(sr_results(:, a)), ...
        mean(time_results(:, a)) * 1000);
end

fprintf('\n========================================\n');
fprintf('闁槒绶宀冪槈\n');
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
    fprintf('閴?Exhaustive 閺堝娓舵担搴ｆ畱 QoE Cost (%.3f) - Ground Truth\n', min_qoe);
else
    fprintf('閳?%s (%.3f) 娴ｅ簼绨?Exhaustive (%.3f)\n', ...
        algorithms{min_idx}, min_qoe, mean_qoe(idx_exhaustive));
end

% Check 2: Proposed should be close to Exhaustive
gap_proposed = (mean_qoe(idx_proposed) - mean_qoe(idx_exhaustive)) / mean_qoe(idx_exhaustive) * 100;
fprintf('  Proposed 娑?Exhaustive 瀹割喛绐? %.1f%%\n', gap_proposed);
if gap_proposed < 5
    fprintf('閴?Proposed 闂堢偛鐖堕幒銉ㄧ箮閺堚偓娴兼﹫绱抃n');
elseif gap_proposed < 15
    fprintf('閴?Proposed 閹恒儴绻庨張鈧导姗堢礄閸欘垱甯撮崣妤嬬礆\n');
else
    fprintf('閳?Proposed 娑撳孩娓舵导妯烘▕鐠烘繆绶濇径顪');
end

% Check 3: Proposed vs GA (time-fair comparison)
if mean_qoe(idx_proposed) < mean_qoe(idx_ga)
    improvement = (mean_qoe(idx_ga) - mean_qoe(idx_proposed)) / mean_qoe(idx_ga) * 100;
    fprintf('閴?Proposed (%.3f) 濮?GA (%.3f) 婵?%.1f%% - 閺冨爼妫块崗顒€閽╃€佃鐦稉瀣箯閼虫粣绱抃n', ...
        mean_qoe(idx_proposed), mean_qoe(idx_ga), improvement);
elseif abs(mean_qoe(idx_proposed) - mean_qoe(idx_ga)) / mean_qoe(idx_ga) < 0.05
    fprintf('閳?Proposed (%.3f) 娑?GA (%.3f) 閻╃缍媆n', ...
        mean_qoe(idx_proposed), mean_qoe(idx_ga));
else
    fprintf('閳?GA (%.3f) 閻ｃ儰绱禍?Proposed (%.3f)\n', ...
        mean_qoe(idx_ga), mean_qoe(idx_proposed));
end

% Check 4: Proposed vs Norm
if mean_qoe(idx_proposed) < mean_qoe(idx_norm)
    improvement = (mean_qoe(idx_norm) - mean_qoe(idx_proposed)) / mean_qoe(idx_norm) * 100;
    fprintf('閴?Proposed (%.3f) 濮?Norm (%.3f) 婵?%.1f%%\n', ...
        mean_qoe(idx_proposed), mean_qoe(idx_norm), improvement);
else
    fprintf('閳?Proposed 娑?Norm 瀹割喛绐涙稉宥嗘閺勭斗n');
end

% Check 5: Time comparison
mean_time = mean(time_results, 1) * 1000;  % ms
fprintf('\n鏉╂劘顢戦弮鍫曟？鐎佃鐦?\n');
fprintf('  Proposed: %.1f ms\n', mean_time(idx_proposed));
fprintf('  GA:       %.1f ms\n', mean_time(idx_ga));
fprintf('  Exhaustive: %.1f ms\n', mean_time(idx_exhaustive));
if mean_time(idx_proposed) < mean_time(idx_ga)
    fprintf('閴?Proposed 濮?GA 韫?%.1fx\n', mean_time(idx_ga)/mean_time(idx_proposed));
end

% Check 6: Random should be worst
if mean_qoe(idx_random) == max(mean_qoe)
    fprintf('閴?Random 閻?QoE Cost 閺堚偓妤?(%.3f) - 閸╄櫣鍤庡锝団€榎n', mean_qoe(idx_random));
end

fprintf('\n========================================\n');
fprintf('妤犲矁鐦夌€瑰本鍨歕n');
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

function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights(:).', cfg.num_users, 1);
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
end

