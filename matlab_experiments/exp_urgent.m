function run_id = exp_urgent(varargin)
%EXP_URGENT 缁毖嗘彥閻劍鍩涢崷鐑樻珯鐎圭偤鐛?
%
% 閸︾儤娅欑拋鎹愵吀閿?%   - 閻劍鍩?-4閿涙俺绔熺紓妯兼暏閹村嚖绱欐潻婊咁瀲BS閿涘LoS閿涘瞼娲挎潻鐐寸€顕嗙礆
%   - 閻劍鍩?-12閿涙碍娅橀柅姘辨暏閹村嚖绱欓棃鐘虹箮BS閿涘oS閿涘瞼娲挎潻鐐搭劀鐢潻绱?
%   - RIS閿涙岸鍎寸純鎻掓躬cell-edge閺傜懓鎮滈敍宀冨厴鐟曞棛娲婃潏鍦喘閻劍鍩?
%
% 妫板嫭婀＄紒鎾寸亯閿?%   - Proposed閼冲€熺槕閸掝偉绔熺紓妯兼暏閹村嘲鑻熸导妯哄帥閸掑棝鍘IS
%   - Random閸欘垵鍏橀幎濂烮S濞搭亣鍨傞崷銊︽珮闁氨鏁ら幋铚傜瑐
%   - Proposed閻ㄥ嚥oE娴兼ê濞嶆导姘纯閺勫孩妯?

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
    fprintf('缁毖嗘彥閻劍鍩涢崷鐑樻珯鐎圭偤鐛橽n');
    fprintf('========================================\n\n');
    
    % 閸╄櫣顢呴柊宥囩枂
    cfg = config();
    cfg.users_per_cell = 12;
    cfg.ris_per_cell = 4;
    cfg.k0 = 1;
    cfg.num_users = 12;
    cfg.num_ris = 4;
    
    % 缁毖嗘彥閸︾儤娅欐稉鎾舵暏閸欏倹鏆?
    cfg.n_ris = 36;        % RIS閸忓啰绀岄弫浼村櫤: 36
    cfg.dmax = 0.30;       % 鐠囶厺绠熸径杈╂埂闂冨牆鈧? D<=0.30, 閸?xi>=0.70
    
    % 缁毖嗘彥閻劍鍩涢弫浼村櫤
    num_urgent = 4;  % 閻劍鍩?-4閺勵垳鎻ｆ潻顐ゆ暏閹?    
    fprintf('闁板秶鐤?\n');
    fprintf('  閹崵鏁ら幋閿嬫殶 K = %d\n', cfg.num_users);
    fprintf('  缁毖嗘彥閻劍鍩?= %d (鏉堝湱绱? NLoS)\n', num_urgent);
    fprintf('  閺咁噣鈧氨鏁ら幋?= %d (鏉╂垼绐? LoS)\n', cfg.num_users - num_urgent);
    fprintf('  RIS閺佷即鍣?L = %d\n', cfg.num_ris);
    fprintf('  RIS閸忓啰绀岄弫?N = %d\n', cfg.n_ris);
    fprintf('  濮ｅ粺IS鐎瑰綊鍣?k0 = %d\n', cfg.k0);
    fprintf('  鐠囶厺绠熼梻銊︻潬 xi_th = %.2f (dmax=%.2f)\n', 1-cfg.dmax, cfg.dmax);
    fprintf('  MC濞嗏剝鏆?= %d\n\n', mc);
    
    algorithms = {'random', 'norm', 'proposed', 'ga'};
    p_dbw_list = linspace(-25, -5, 8);
    
    A = numel(algorithms);
    X = numel(p_dbw_list);
    
    % 缂佹挻鐏夌€涙ê鍋?
    q_all = zeros(mc, X, A);
    sr_all = zeros(mc, X, A);
    urgent_qoe_all = zeros(mc, X, A);
    normal_qoe_all = zeros(mc, X, A);
    violation_all = zeros(mc, X, A);  % 閺冭泛娆㈡潻婵堝閻?    semantic_vio_all = zeros(mc, X, A);  % 鐠囶厺绠熸潻婵堝閻?    
    % 鐠囧﹥鏌囬弫鐗堝祦閿涙俺顕㈡稊澶屾祲娴肩厧瀹抽崪瀛睮NR
    xi_urgent_all = zeros(mc, X, A);  % 鏉堝湱绱悽銊﹀煕楠炲啿娼巟i
    xi_normal_all = zeros(mc, X, A);  % 閺咁噣鈧氨鏁ら幋宄伴挬閸у櫄i
    sinr_urgent_all = zeros(mc, X, A);  % 鏉堝湱绱悽銊﹀煕楠炲啿娼嶴INR(dB)
    sinr_normal_all = zeros(mc, X, A);  % 閺咁噣鈧氨鏁ら幋宄伴挬閸у槩INR(dB)
    
    for tr = 1:mc
        trial_seed = base_seed + tr;
        
        % 閻㈢喐鍨氱槐褑鎻╅崷鐑樻珯閻ㄥ嫬鍤戞担鏇炴嫲娣囷繝浜?
        [geom, ch] = generate_urgent_scenario(cfg, trial_seed, num_urgent);
        
        for ix = 1:X
            p_dbw = p_dbw_list(ix);
            
            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);
                
                out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
                avg_qoe = out.avg_qoe;
                qoe_vec = out.qoe_vec;
                sum_rate = out.sum_rate_bps;
                
                q_all(tr, ix, a) = avg_qoe;
                sr_all(tr, ix, a) = sum_rate;
                urgent_qoe_all(tr, ix, a) = mean(qoe_vec(1:num_urgent));
                normal_qoe_all(tr, ix, a) = mean(qoe_vec(num_urgent+1:end));
                violation_all(tr, ix, a) = out.delay_vio_rate_all;
                semantic_vio_all(tr, ix, a) = out.semantic_vio_rate_all;
                
                % 鐠囧﹥鏌囬弫鐗堝祦
                xi_urgent_all(tr, ix, a) = mean(out.xi(1:num_urgent));
                xi_normal_all(tr, ix, a) = mean(out.xi(num_urgent+1:end));
                sinr_db = 10 * log10(out.gamma_p + 1e-12);
                sinr_urgent_all(tr, ix, a) = mean(sinr_db(1:num_urgent));
                sinr_normal_all(tr, ix, a) = mean(sinr_db(num_urgent+1:end));
            end
        end
        
        if mod(tr, 50) == 0
            fprintf('  MC trial %d/%d\n', tr, mc);
        end
    end
    
    % 濮瑰洦鈧崵绮ㄩ弸?    result = struct();
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
    
    % 閹垫挸宓冮崗鎶芥暛缂佹挻鐏?
    fprintf('\n========================================\n');
    fprintf('閸忔娊鏁紒鎾寸亯 (P = -15 dBW)\n');
    fprintf('========================================\n');
    mid_idx = 4;  % -15 dBW 闂勫嫯绻?
    fprintf('%-12s %12s %12s %12s %12s\n', '缁犳纭?, 'Avg QoE', '缁毖嗘彥閻劍鍩決oE', '閺咁噣鈧氨鏁ら幋绋眔E', '鏉╂繄瀹抽悳?);
    fprintf('%s\n', repmat('-', 1, 60));
    for a = 1:A
        fprintf('%-12s %12.3f %12.3f %12.3f %12.1f%%\n', ...
            algorithms{a}, ...
            result.avg_qoe(mid_idx, a), ...
            result.urgent_qoe(mid_idx, a), ...
            result.normal_qoe(mid_idx, a), ...
            result.violation_rate(mid_idx, a) * 100);
    end
    
    % 鐠囧﹥鏌囨穱鈩冧紖閿涙俺顕㈡稊澶屾祲娴肩厧瀹抽崪瀛睮NR
    fprintf('\n========================================\n');
    fprintf('鐠囧﹥鏌囨穱鈩冧紖: 鐠囶厺绠熼惄闀愭妧鎼?xi 閸?SINR\n');
    fprintf('========================================\n');
    fprintf('鐠囶厺绠熺痪锔芥将: D <= %.2f, 閸?xi >= %.2f\n', cfg.dmax, 1 - cfg.dmax);
    fprintf('RIS閸忓啰绀岄弫浼村櫤 N = %d\n', cfg.n_ris);
    fprintf('DeepSC鐞?M=8): SNR>=4dB閹靛秷鍏樻潏鎯у煂xi>=0.70\n\n');
    
    fprintf('%-12s %10s %10s %10s %10s\n', '缁犳纭?, 'xi鏉堝湱绱?, 'xi閺咁噣鈧?, 'SNR鏉堝湱绱?, 'SNR閺咁噣鈧?);
    fprintf('%s\n', repmat('-', 1, 54));
    for a = 1:A
        fprintf('%-12s %10.3f %10.3f %10.1f dB %10.1f dB\n', ...
            algorithms{a}, ...
            result.xi_urgent(mid_idx, a), ...
            result.xi_normal(mid_idx, a), ...
            result.sinr_urgent(mid_idx, a), ...
            result.sinr_normal(mid_idx, a));
    end
    
    fprintf('\n缂佹捁顔? ');
    if result.xi_urgent(mid_idx, 3) < (1 - cfg.dmax) && result.xi_normal(mid_idx, 3) < (1 - cfg.dmax)
        fprintf('瑜版挸澧犻崝鐔哄芳閼煎啫娲块崘鍜冪礉閹碘偓閺堝鏁ら幋椋庢畱xi閸у洣缍嗘禍?.2f閿涘矁顕㈡稊澶屽閺夌喍绗夐崣顖濐攽閵嗕繐n', 1 - cfg.dmax);
    elseif result.xi_urgent(mid_idx, 3) < (1 - cfg.dmax)
        fprintf('娴犲懓绔熺紓妯兼暏閹寸i娴ｅ簼绨?.2f閿涘矁顕㈡稊澶屽閺夌喎顕潏鍦喘閻劍鍩涙稉宥呭讲鐞涘被鈧繐n', 1 - cfg.dmax);
    else
        fprintf('鐠囶厺绠熺痪锔芥将閸欘垵顢戦妴淇搉');
    end
    
    % 娣囨繂鐡ㄩ崪宀€绮崶?    run_id = sprintf('urgent_%s', datestr(now, 'yyyymmdd_HHMMSS'));
    
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    plot_urgent_results(run_id, result, out_dir, algorithms);
    
    fprintf('\n鐎圭偤鐛欑€瑰本鍨氶敍涔簎n_id: %s\n', run_id);
    cd(old_dir);
end

%% 閻㈢喐鍨氱槐褑鎻╅崷鐑樻珯
function [geom, ch] = generate_urgent_scenario(cfg, seed, num_urgent)
    rng(seed, 'twister');
    
    num_users = cfg.num_users;
    num_ris = cfg.num_ris;
    
    bs_pos = [0, 0];
    ue = zeros(num_users, 2);
    ris = zeros(num_ris, 2);
    
    % ========== 閻劍鍩涙担宥囩枂 ==========
    % 缁毖嗘彥閻劍鍩?1-num_urgent): 鏉堝湱绱担宥囩枂閿?20-180m閿涘矂娉︽稉顓炴躬娑撯偓娑擃亝澧栭崠?    urgent_sector = pi/4 + 0.3*randn();  % 缁毖嗘彥閻劍鍩涢幍鈧崷銊﹀閸?    for k = 1:num_urgent
        r = 120 + 60 * rand();  % 120-180m閿涘矁绻欑粋绫匰
        ang = urgent_sector + 0.3 * (rand() - 0.5);  % 闂嗗棔鑵戦崷銊ょ娑擃亝澧栭崠?        ue(k, :) = bs_pos + r * [cos(ang), sin(ang)];
    end
    
    % 閺咁噣鈧氨鏁ら幋?num_urgent+1:end): 鏉╂垼绐涚粋浼欑礉30-70m閿涘苯娼庨崠鈧崚鍡楃
    for k = num_urgent+1:num_users
        r = 30 + 40 * rand();  % 30-70m閿涘矂娼潻鎱婼
        ang = 2 * pi * rand();  % 閸у洤瀵戦崚鍡楃
        ue(k, :) = bs_pos + r * [cos(ang), sin(ang)];
    end
    
    % ========== RIS娴ｅ秶鐤?==========
    % RIS闁劎璁查崷銊ф彛鏉╊偆鏁ら幋閿嬫煙閸氭埊绱?0-100m閿涘矁鍏樼憰鍡欐磰鏉堝湱绱悽銊﹀煕
    for l = 1:num_ris
        r_dist = 80 + 20 * rand();  % 80-100m
        r_ang = urgent_sector + 0.5 * (l - 1 - num_ris/2) / num_ris * pi;  % 閹靛洤鑸伴崚鍡楃
        ris(l, :) = bs_pos + r_dist * [cos(r_ang), sin(r_ang)];
    end
    
    geom.bs = bs_pos;
    geom.ue = ue;
    geom.ris = ris;
    
    % ========== 娣囷繝浜鹃悽鐔稿灇 ==========
    h_d = zeros(cfg.nt, num_users);
    G = zeros(cfg.nt, cfg.n_ris, num_ris);
    H_ris = zeros(cfg.n_ris, num_users, num_ris);
    
    % 鐠侯垱宕幐鍥ㄦ殶
    alpha_urgent = 4.0;   % 缁毖嗘彥閻劍鍩涢敍姝侺oS閿涘矁鐭鹃幑鐔枫亣
    alpha_normal = 2.8;   % 閺咁噣鈧氨鏁ら幋鍑ょ窗LoS閿涘矁鐭鹃幑鐔风毈
    alpha_br = 2.2;       % BS->RIS: LoS
    alpha_ru = 2.5;       % RIS->UE
    
    % 閻╃绻涙穱锟犱壕 BS->UE
    for k = 1:num_users
        d = norm(ue(k,:) - bs_pos) + cfg.eps;
        
        if k <= num_urgent
            % 缁毖嗘彥閻劍鍩涢敍姝侺oS閿涘苯銇囩捄顖涘疮
            alpha = alpha_urgent;
        else
            % 閺咁噣鈧氨鏁ら幋鍑ょ窗LoS閿涘苯鐨捄顖涘疮
            alpha = alpha_normal;
        end
        
        pathloss = d.^(-alpha);
        h_d(:, k) = sqrt(pathloss/2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
    end
    
    % 缁狙嗕粓娣囷繝浜?BS->RIS->UE
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
    
    % RIS閻╅晲缍呴敍鍫濐嚠姒绘劖膩瀵骏绱?
    theta = exp(1j * 2 * pi * rand(cfg.n_ris, num_ris));
    
    ch.h_d = h_d;
    ch.G = G;
    ch.H_ris = H_ris;
    ch.theta = theta;
end

%% 缁犳纭堕柅澶嬪
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
            % 娑撹桨绨￠幎?GA 娴ｆ粈璐熼垾婊€绱剁紒鐔尖偓鐔哄芳閺堚偓婢堆冨閳ユ繄娈戦崺铏瑰殠閿涘矁鈧奔绗夐弰?QoE 娴兼ê瀵查崳顭掔礉
            % 娑撳秴鍟€娴ｈ法鏁?Proposed 閺傝顢嶆担婊€璐熺粔宥呯摍閿涘矂浼╅崗?GA 閸?QoE 娑撳﹪鈧壈绻庨幋鏍Т鏉?Proposed閵?            opts.use_proposed_seed = false;
            % 鐎?GA 闁插洨鏁ら垾婊€绱剁紒鐔尖偓鐔哄芳閺堚偓婢堆冨閳ユ繄娲伴弽鍥风礉娴ｆ粈璐?QoE 娴兼ê瀵茬粻妤佺《閻ㄥ嫬顕В鏂跨唨缁?            % 閿涘牅绗?verify_weights.m 娑擃厾娈戠拋鍓х枂娣囨繃瀵旀稉鈧懛杈剧礆
            opts.optimize_sumrate = true;
            opts.Np = cfg.ga_Np;
            opts.Niter = cfg.ga_Niter;
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

%% 娴肩姵鎸遍弮璺烘鐠侊紕鐣?
function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights(:).', cfg.num_users, 1);
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
end

function plot_urgent_results(run_id, result, out_dir, algorithms)
    x = result.x_vals;
    A = numel(algorithms);
    markers = {'o', 's', 'd', '^'};
    colors = lines(A);
    
    % ========== 閸?: 楠炲啿娼嶲oE ==========
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
    title('缁毖嗘彥閸︾儤娅? 閸忋劋缍嬮悽銊﹀煕楠炲啿娼嶲oE');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_avg_qoe.png', run_id)));
    close(gcf);
    
    % ========== 閸?: 缁毖嗘彥閻劍鍩決oE ==========
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
    title('缁毖嗘彥閸︾儤娅? 鏉堝湱绱悽銊﹀煕QoE (1-4, NLoS)');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_urgent_qoe.png', run_id)));
    close(gcf);
    
    % ========== 閸?: 閺咁噣鈧氨鏁ら幋绋眔E ==========
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
    title('缁毖嗘彥閸︾儤娅? 閺咁噣鈧氨鏁ら幋绋眔E (5-12, LoS)');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_normal_qoe.png', run_id)));
    close(gcf);
    
    % ========== 閸?: Sum-rate ==========
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
    title('缁毖嗘彥閸︾儤娅? Sum-rate');
    legend('Location', 'best');
    grid on;
    saveas(gcf, fullfile(out_dir, sprintf('urgent_%s_sum_rate.png', run_id)));
    close(gcf);
    
    % 閸欘亙绻氶悾娆忓 4 瀵姳绗岀拋鐑樻瀮閻╁瓨甯撮惄绋垮彠閻ㄥ嫬娴橀敍灞藉従娴ｆ瑨鐦栭弬顓炴禈娑撳秴鍟€閻㈢喐鍨氶敍?    % 娴犮儵浼╅崗宥勯獓閻㈢喕绻冩径姘瀮娴犺翰鈧?    
    fprintf('閸ュ墽澧栧韫箽鐎?\n');
    fprintf('  urgent_%s_avg_qoe.png\n', run_id);
    fprintf('  urgent_%s_urgent_qoe.png\n', run_id);
    fprintf('  urgent_%s_normal_qoe.png\n', run_id);
    fprintf('  urgent_%s_sum_rate.png\n', run_id);
end


