function run_id = sweep_ris_count(varargin)
%SWEEP_RIS_COUNT 閸︺劎鎻ｆ潻顐㈡簚閺咁垯绗呴幍顐ｅ伎 RIS 閺佷即鍣洪敍鍫滅瑢 exp_urgent 閸欙絽绶炴稉鈧懛杈剧礆
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
    addParameter(p, 'p_dbw', -15);
    addParameter(p, 'L_list', [1 2 3 4 6 8]);
    addParameter(p, 'num_urgent', 4);
    parse(p, varargin{:});

    mc = p.Results.mc;
    base_seed = p.Results.seed;
    p_dbw = p.Results.p_dbw;
    L_list = p.Results.L_list;
    num_urgent = p.Results.num_urgent;

    cfg0 = config();
    cfg0.users_per_cell = 12;
    cfg0.num_users = 12;
    cfg0.k0 = 1;
    cfg0.n_ris = 36;
    cfg0.dmax = 0.30;

    algorithms = {'random', 'norm', 'proposed', 'ga'};
    A = numel(algorithms);
    X = numel(L_list);

    q_avg_all = zeros(mc, X, A);
    q_urgent_all = zeros(mc, X, A);
    q_normal_all = zeros(mc, X, A);
    sum_rate_all = zeros(mc, X, A);

    fprintf('========================================\n');
    fprintf('SWEEP RIS COUNT (Urgent Scenario)\n');
    fprintf('========================================\n');
    fprintf('MC=%d, p=%.1f dBW, L=%s\n', mc, p_dbw, mat2str(L_list));

    maxL = max(L_list);

    for tr = 1:mc
        trial_seed = base_seed + tr;

        % 閸忓牏鏁撻幋鎰付婢?L 閻ㄥ嫮鎻ｆ潻顐㈡簚閺咁垽绱濋崘宥堫梿閸擃亜澧?L 娑?RIS閿涘奔绻氱拠浣风瑝閸?L 閸欘垱鐦?
        cfg_max = cfg0;
        cfg_max.ris_per_cell = maxL;
        cfg_max.num_ris = maxL;
        [geom_full, ch_full] = generate_urgent_scenario(cfg_max, trial_seed, num_urgent);

        for ix = 1:X
            L = L_list(ix);

            cfg = cfg0;
            cfg.ris_per_cell = L;
            cfg.num_ris = L;

            geom = geom_full;
            geom.ris = geom_full.ris(1:L, :);

            ch = ch_full;
            ch.G = ch_full.G(:, :, 1:L);
            ch.H_ris = ch_full.H_ris(:, :, 1:L);
            ch.theta = ch_full.theta(:, 1:L);

            for a = 1:A
                alg = algorithms{a};
                assign = pick_assignment(cfg, ch, alg, p_dbw, trial_seed, geom);

                out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, cfg.weights(1,:));
                qoe_vec = out.qoe_vec;
                sum_rate = out.sum_rate_bps;

                if any(~isfinite(qoe_vec)) || any(~isfinite(gamma)) || ~isfinite(sum_rate)
                    error('Non-finite metric: trial=%d, L=%d, alg=%s', tr, L, alg);
                end

                q_avg_all(tr, ix, a) = mean(qoe_vec);
                q_urgent_all(tr, ix, a) = mean(qoe_vec(1:num_urgent));
                q_normal_all(tr, ix, a) = mean(qoe_vec(num_urgent+1:end));
                sum_rate_all(tr, ix, a) = sum_rate;
            end
        end

        if mod(tr, 20) == 0
            fprintf('  MC %d/%d\n', tr, mc);
        end
    end

    result = struct();
    result.algorithms = algorithms;
    result.L_list = L_list;
    result.q_avg = squeeze(mean(q_avg_all, 1));
    result.q_urgent = squeeze(mean(q_urgent_all, 1));
    result.q_normal = squeeze(mean(q_normal_all, 1));
    result.sum_rate = squeeze(mean(sum_rate_all, 1));
    result.avg_qoe = result.q_avg;

    run_id = save_results('sweep_ris_count', result, L_list);
    plot_ris_figs(result, proj_root, run_id);

    fprintf('鐎瑰本鍨氶敍姝硊n_id=%s\n', run_id);
    cd(old_dir);
end

function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights(:).', cfg.num_users, 1);
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
end

function [geom, ch] = generate_urgent_scenario(cfg, seed, num_urgent)
    rng(seed, 'twister');

    num_users = cfg.num_users;
    num_ris = cfg.num_ris;

    bs_pos = [0, 0];
    ue = zeros(num_users, 2);
    ris = zeros(num_ris, 2);

    % 缁毖嗘彥閻劍鍩? 鏉堝湱绱妴涓疞oS 閸婃儳鎮?
    urgent_sector = pi/4 + 0.3 * randn();
    for k = 1:num_urgent
        r = 120 + 60 * rand();
        ang = urgent_sector + 0.3 * (rand() - 0.5);
        ue(k, :) = bs_pos + r * [cos(ang), sin(ang)];
    end

    % 閺咁噣鈧氨鏁ら幋? 鏉╂垼绐涢妴涓﹐S 閸婃儳鎮?
    for k = num_urgent+1:num_users
        r = 30 + 40 * rand();
        ang = 2 * pi * rand();
        ue(k, :) = bs_pos + r * [cos(ang), sin(ang)];
    end

    % RIS 闁劎璁查崷銊ㄧ珶缂傛鏁ら幋閿嬫煙閸?    for l = 1:num_ris
        r_dist = 80 + 20 * rand();
        r_ang = urgent_sector + 0.5 * (l - 1 - num_ris/2) / num_ris * pi;
        ris(l, :) = bs_pos + r_dist * [cos(r_ang), sin(r_ang)];
    end

    geom.bs = bs_pos;
    geom.ue = ue;
    geom.ris = ris;

    h_d = zeros(cfg.nt, num_users);
    G = zeros(cfg.nt, cfg.n_ris, num_ris);
    H_ris = zeros(cfg.n_ris, num_users, num_ris);

    alpha_urgent = 4.0;
    alpha_normal = 2.8;
    alpha_br = 2.2;
    alpha_ru = 2.5;

    for k = 1:num_users
        d = norm(ue(k, :) - bs_pos) + cfg.eps;
        if k <= num_urgent
            alpha = alpha_urgent;
        else
            alpha = alpha_normal;
        end
        pathloss = d.^(-alpha);
        h_d(:, k) = sqrt(pathloss/2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
    end

    for l = 1:num_ris
        d_br = norm(ris(l, :) - bs_pos) + cfg.eps;
        pl_br = d_br.^(-alpha_br);
        G(:, :, l) = sqrt(pl_br/2) * (randn(cfg.nt, cfg.n_ris) + 1j * randn(cfg.nt, cfg.n_ris));

        for k = 1:num_users
            d_ru = norm(ue(k, :) - ris(l, :)) + cfg.eps;
            pl_ru = d_ru.^(-alpha_ru);
            H_ris(:, k, l) = sqrt(pl_ru/2) * (randn(cfg.n_ris, 1) + 1j * randn(cfg.n_ris, 1));
        end
    end

    theta = exp(1j * 2 * pi * rand(cfg.n_ris, num_ris));

    ch.h_d = h_d;
    ch.G = G;
    ch.H_ris = H_ris;
    ch.theta = theta;
end

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

function plot_ris_figs(result, proj_root, run_id)
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir')
        mkdir(out_dir);
    end

    L = result.L_list;
    algs = result.algorithms;
    colors = lines(numel(algs));
    markers = {'o', 's', 'd', '^'};

    configs = {
        'q_urgent', 'RIS 閺佷即鍣烘晶鐐插: 缁毖嗘彥閻劍鍩?QoE Cost (1-4)', 'Urgent Users QoE Cost';
        'sum_rate', 'RIS 閺佷即鍣烘晶鐐插: 缁崵绮?Sum-rate', 'Sum-rate (bps)';
        'q_normal', 'RIS 閺佷即鍣烘晶鐐插: 閺咁噣鈧氨鏁ら幋?QoE Cost (5-12)', 'Normal Users QoE Cost';
        'q_avg',    'RIS 閺佷即鍣烘晶鐐插: 閸忋劋缍嬮悽銊﹀煕楠炲啿娼?QoE Cost', 'Avg QoE Cost'
    };

    for i = 1:size(configs, 1)
        figure('Color', 'w', 'Position', [120 120 1100 700]);
        hold on;
        data = result.(configs{i,1});
        for a = 1:numel(algs)
            plot(L, data(:, a), ['-' markers{a}], ...
                'Color', colors(a,:), 'LineWidth', 1.8, 'MarkerSize', 9, ...
                'DisplayName', algs{a});
        end
        hold off;
        title(configs{i,2});
        xlabel('RIS 閺佷即鍣?L');
        ylabel(configs{i,3});
        grid on;
        legend('Location', 'best');

        saveas(gcf, fullfile(out_dir, sprintf('%s_vs_L_%s.png', configs{i,1}, run_id)));
        close(gcf);
    end
end


