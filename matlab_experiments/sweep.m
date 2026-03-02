function run_id = sweep(varargin)
%SWEEP 閻㈢喐鍨氱拋鐑樻瀮鐟曚焦鐪伴惃?4 瀵姴濮涢悳鍥閸斻劌娴?% 1. 缁毖嗘彥閻劍鍩?(1-4) QoE Cost vs p
% 2. 閺咁噣鈧氨鏁ら幋?(5-12) QoE Cost vs p
% 3. 閸忋劋缍嬮悽銊﹀煕楠炲啿娼?QoE Cost vs p
% 4. 缁崵绮?Sum-rate vs p

    this_file = mfilename('fullpath');
    this_dir = fileparts(this_file);
    proj_root = fileparts(this_dir);
    
    addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
    addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
    addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');
    
    old_dir = cd(proj_root);

    cfg = config();
    cfg = apply_overrides(cfg, varargin{:});

    if ~isfield(cfg,'mc') || isempty(cfg.mc), cfg.mc = 200; end
    if ~isfield(cfg,'seed') || isempty(cfg.seed), cfg.seed = 42; end
    if ~isfield(cfg,'p_dbw_list') || isempty(cfg.p_dbw_list)
        cfg.p_dbw_list = linspace(-25, -5, 8);
    end

    algorithms = {'random', 'norm', 'proposed', 'ga'};
    A = numel(algorithms);

    x_vals = cfg.p_dbw_list(:).';
    X = numel(x_vals);
    mc = cfg.mc;

    fprintf('========================================\n');
    fprintf('SWEEP Experiment (Strict & Fixed Eval)\n');
    fprintf('========================================\n');
    fprintf('Power: [%.0f, %.0f] dBW, MC=%d\n', x_vals(1), x_vals(end), mc);

    result = struct();
    result.algorithms = algorithms;
    result.x_vals = x_vals;
    result.mc = mc;

    % 缂佺喕顓搁惌鈺呮█ [MC x X x A]
    sr_all = zeros(mc, X, A);
    q_avg_all = zeros(mc, X, A);
    q_urgent_all = zeros(mc, X, A);
    q_normal_all = zeros(mc, X, A);

    for tr = 1:mc
        trial_seed = cfg.seed + tr;
        % --- 閸忔娊鏁穱顔碱槻閿涙岸鏀ｇ€规艾婧€閺?---
        rng(trial_seed, 'twister');
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);
        profile = build_profile_urgent_normal(cfg, geom, struct());
        [urgent_idx, normal_idx] = get_group_indices(cfg, profile);

        for ix = 1:X
            p_dbw = x_vals(ix);
            for a = 1:A
                alg = algorithms{a};
                
                if strcmpi(alg, 'proposed')
                    % Proposed: use full AO output for evaluation.
                    [assign, theta_opt, beam_opt, ~] = ua_qoe_ao(cfg, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, cfg.weights(1,:));
                    sol = struct('assign', assign, 'theta_all', theta_opt, 'V', beam_opt);
                    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
                else
                    assign = pick_assignment(cfg, ch, geom, alg, p_dbw, trial_seed);
                    h_eff = compute_h_eff(cfg, ch, assign, ch.theta);
                    % Run WMMSE for baseline fairness.
                    [V_base, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users,1), 5);
                    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V_base);
                    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct());
                end

                sr_all(tr, ix, a) = out.sum_rate_bps;
                q_avg_all(tr, ix, a) = mean(out.qoe_vec);
                q_urgent_all(tr, ix, a) = mean(out.qoe_vec(urgent_idx));
                q_normal_all(tr, ix, a) = mean(out.qoe_vec(normal_idx));
            end
        end
        if mod(tr, 20) == 0
            fprintf('  MC trial %d/%d\n', tr, mc);
        end
    end

    result.sum_rate = squeeze(mean(sr_all, 1));
    result.q_avg = squeeze(mean(q_avg_all, 1));
    result.q_urgent = squeeze(mean(q_urgent_all, 1));
    result.q_normal = squeeze(mean(q_normal_all, 1));
    result.avg_qoe = result.q_avg;

    if ~exist('results', 'dir'), mkdir('results'); end
    if ~exist('figures', 'dir'), mkdir('figures'); end

    % 娣囨繂鐡ㄩ弫鐗堝祦
    run_id = datestr(now,'yyyymmdd_HHMMSS');
    save(fullfile('results', ['sweep_strict_' run_id '.mat']), 'result');

    plot_power_figs(result, proj_root, run_id);

    fprintf('\nSweep complete. run_id: %s\n', run_id);
    cd(old_dir);
end

function cfg = apply_overrides(cfg, varargin)
    if isempty(varargin), return; end
    if mod(numel(varargin),2) ~= 0, error('Overrides must be key-value pairs.'); end
    for i = 1:2:numel(varargin)
        k = char(varargin{i}); v = varargin{i+1};
        cfg.(k) = v;
    end
end

function assign = pick_assignment(cfg, ch, geom, alg, p_dbw, seed)
    if nargin < 6, seed = 1; end
    w = cfg.weights(1,:);
    semantic_mode = cfg.semantic_mode;
    table_path = cfg.semantic_table;
    switch alg
        case 'random'
            assign = random_match(cfg, seed);
        case 'norm'
            assign = norm_based(cfg, ch);
        case 'ga'
            opts.Np = cfg.ga_Np; opts.Niter = cfg.ga_Niter;
            opts.geom = geom; opts.weights = w;
            opts.semantic_mode = semantic_mode; opts.table_path = table_path;
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
        otherwise
            assign = norm_based(cfg, ch);
    end
end

function h_eff = compute_h_eff(cfg, ch, assign, theta_all)
    h_eff = ch.h_d;
    for k = 1:cfg.num_users
        l = assign(k);
        if l > 0 && l <= cfg.num_ris
            h_eff(:, k) = h_eff(:, k) + cfg.ris_gain * ch.G(:,:,l) * (theta_all(:,l) .* ch.H_ris(:,k,l));
        end
    end
end

function plot_power_figs(result, proj_root, run_id)
    out_dir = fullfile(proj_root, 'figures');
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    x = result.x_vals; algs = result.algorithms; A = numel(algs);
    colors = lines(A); markers = {'o', 's', 'd', '^'};
    fig_configs = {
        'q_urgent', '缁毖嗘彥閸︾儤娅欓敍姘崇珶缂傛鏁ら幋绋眔E (1-4, NLoS)', 'Urgent Users QoE Cost', sprintf('urgent_qoe_%s.png', run_id);
        'q_normal', '缁毖嗘彥閸︾儤娅欓敍姘珮闁氨鏁ら幋绋眔E (5-12, LoS)', 'Normal Users QoE Cost', sprintf('normal_qoe_%s.png', run_id);
        'q_avg',    '缁毖嗘彥閸︾儤娅欓敍姘弿娴ｆ挾鏁ら幋宄伴挬閸у槧oE',        'Avg QoE Cost',          sprintf('avg_qoe_%s.png', run_id);
        'sum_rate', '缁毖嗘彥閸︾儤娅欓敍姝媢m-rate',              'Sum-rate (bps)',        sprintf('sum_rate_%s.png', run_id)
    };
    for i = 1:size(fig_configs, 1)
        figure('Color', 'w'); hold on;
        data = result.(fig_configs{i, 1});
        % 鏉炶浜曢獮铏拨閿涘牊閮ㄩ崝鐔哄芳鏉炴潙浠?2 閻愯绮﹂崝銊ラ挬閸у浄绱氶敍灞惧絹閸楀洩顫囬幇?        data_smooth = movmean(data, 2, 1);
        for a = 1:numel(algs)
            plot(x, data_smooth(:, a), ['-' markers{a}], 'Color', colors(a,:), 'LineWidth', 1.5, 'MarkerSize', 7, 'DisplayName', algs{a});
        end
        title(fig_configs{i, 2}); xlabel('p (dBW)'); ylabel(fig_configs{i, 3});
        legend('Location', 'best'); grid on;
        saveas(gcf, fullfile(out_dir, fig_configs{i, 4}));
        close(gcf);
    end
end

function [urgent_idx, normal_idx] = get_group_indices(cfg, profile)
K = cfg.num_users;
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('sweep:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required for unified grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
if ~isempty(intersect(urgent_idx, normal_idx))
    error('sweep:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx(:); normal_idx(:)]);
assert(numel(cover) == K && all(cover(:) == (1:K).'), ...
    'sweep:group_coverage: urgent/normal union must cover all users');
end


