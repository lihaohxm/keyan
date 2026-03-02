function run_id = paper_sweep_power(varargin)
%PAPER_SWEEP_POWER Fair power sweep with aligned evaluation chain.
% Run from repo root:
%   paper_sweep_power('mc',50,'seed',42,'out_name','paper_sweep_power_fixed')
%
% Fairness protocol:
% 1) geom/ch/profile are generated once per trial and reused by all algorithms.
% 2) Each algorithm uses an independent derived RNG seed per (trial, p, alg).
% 3) GA fitness uses the same theta/profile definition as final evaluation.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg = config();

p = inputParser;
addParameter(p, 'mc', []);
addParameter(p, 'seed', 42);
addParameter(p, 'p_list', []);
addParameter(p, 'out_name', 'paper_sweep_power_fixed');
addParameter(p, 'ga_log', false);
addParameter(p, 'cfg_overrides', struct());
addParameter(p, 'save_figures', true);
addParameter(p, 'save_mat', true);
addParameter(p, 'save_csv', true);
addParameter(p, 'include_ga_ub', true);
addParameter(p, 'show_ga_ub', false);
addParameter(p, 'ga_rt_pop_size', 8);
addParameter(p, 'ga_rt_num_generations', 6);
addParameter(p, 'ga_rt_budget_evals', 40);
addParameter(p, 'warm_start', false);
addParameter(p, 'debug_norm', false);
addParameter(p, 'debug_power_chain', false);
parse(p, varargin{:});

mc = p.Results.mc;
if isempty(mc)
    if isfield(cfg, 'mc') && ~isempty(cfg.mc)
        mc = cfg.mc;
    else
        mc = 50;
    end
end
seed = p.Results.seed;
p_list = p.Results.p_list;
if isempty(p_list)
    p_list = cfg.p_dbw_list;
end
p_list = p_list(:).';
out_name = char(p.Results.out_name);
cfg = apply_cfg_overrides(cfg, p.Results.cfg_overrides);

alg_names = {'proposed', 'random', 'norm', 'ga_rt'};
if p.Results.include_ga_ub
    alg_names{end + 1} = 'ga_ub';
end
num_alg = numel(alg_names);
num_p = numel(p_list);
tol_q = 1e-9;

avg_qoe_all = zeros(mc, num_p, num_alg);
urgent_qoe_all = zeros(mc, num_p, num_alg);
normal_qoe_all = zeros(mc, num_p, num_alg);
sum_rate_all = zeros(mc, num_p, num_alg);
urgent_sum_rate_all = zeros(mc, num_p, num_alg);
urgent_avg_rate_all = zeros(mc, num_p, num_alg);
urgent_delay_vio_all = zeros(mc, num_p, num_alg);
urgent_semantic_vio_all = zeros(mc, num_p, num_alg);
Qd_mean_all = zeros(mc, num_p, num_alg);
Qs_mean_all = zeros(mc, num_p, num_alg);
ris_count_all = zeros(mc, num_p, num_alg);
proposed_eval_calls_all = nan(mc, num_p);
ga_rt_eval_calls_all = nan(mc, num_p);
ga_rt_budget_target_all = nan(mc, num_p);
sanity = struct( ...
    'qoe_min', inf, 'qoe_max', -inf, ...
    'Qd_min', inf, 'Qd_max', -inf, ...
    'Qs_min', inf, 'Qs_max', -inf);

opts_eval = struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table);

fprintf('========================================\n');
fprintf('paper_sweep_power: MC=%d, seed=%d, numP=%d\n', mc, seed, num_p);
fprintf('algorithms: %s\n', strjoin(alg_names, ', '));
fprintf('========================================\n');

for trial_idx = 1:mc
    trial_seed = seed + trial_idx;
    rng(trial_seed, 'twister');

    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);
    profile = build_profile_urgent_normal(cfg, geom, struct());
    [urgent_idx, normal_idx] = get_group_indices(cfg, profile);
    ws = struct();
    if p.Results.warm_start
        for iw = 1:num_alg
            ws.(alg_names{iw}) = [];
        end
    end

    for ip = 1:num_p
        p_dbw = p_list(ip);
        proposed_budget = [];
        norm_logged_this_p = false;
        if p.Results.debug_power_chain && trial_idx == 1
            debug_print_power_chain(cfg, trial_idx, ip, p_dbw, urgent_idx);
        end

        for ia = 1:num_alg
            alg = alg_names{ia};
            algo_seed = derive_algo_seed(seed, trial_idx, ip, ia);
            rng(algo_seed, 'twister');
            warm_assign = [];
            if p.Results.warm_start && isfield(ws, alg)
                warm_assign = ws.(alg);
            end

            if strcmpi(alg, 'proposed')
                cfgp = cfg;
                if ~isempty(warm_assign) && numel(warm_assign) == cfg.num_users
                    cfgp.ao_init_assign = warm_assign(:);
                end
                [assign, theta_all, V, ao_log] = ua_qoe_ao(cfgp, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, profile);
                sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
                if isfield(ao_log, 'eval_calls')
                    proposed_budget = ao_log.eval_calls;
                else
                    proposed_budget = 100; % conservative fallback
                end
                proposed_eval_calls_all(trial_idx, ip) = proposed_budget;
                if p.Results.warm_start
                    ws.(alg) = assign(:);
                end
            else
                [assign_fixed, assign_info] = pick_assignment_local(cfg, ch, geom, alg, profile, p_dbw, proposed_budget, p.Results.ga_log, p.Results, warm_assign);
                if p.Results.warm_start
                    ws.(alg) = assign_fixed(:);
                end
                if strcmpi(alg, 'ga_rt')
                    % GA-RT uses real-time constrained GA and must stay in light-solve branch.
                    sol = build_light_solution(cfg, ch, assign_fixed, p_dbw);
                    if isfield(assign_info, 'eval_count')
                        ga_rt_eval_calls_all(trial_idx, ip) = assign_info.eval_count;
                    end
                    if isfield(assign_info, 'budget_target')
                        ga_rt_budget_target_all(trial_idx, ip) = assign_info.budget_target;
                    end
                else
                    [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign_fixed, profile, struct());
                    sol = sol_fixed;
                end
            end

            out = evaluate_system_rsma(cfg, ch, geom, sol, profile, opts_eval);
            rate_vec_bps = get_rate_vec_bps_compat(out, cfg.num_users); % bps
            urgent_sum_rate_bps = sum(rate_vec_bps(urgent_idx)); % bps
            urgent_avg_rate_bps = mean(rate_vec_bps(urgent_idx)); % bps
            sanity = update_sanity_bounds(sanity, out, tol_q, sprintf('trial=%d,p_idx=%d,alg=%s', trial_idx, ip, alg));

            if p.Results.debug_norm && strcmpi(alg, 'norm') && trial_idx == 1 && ~norm_logged_this_p
                debug_print_norm_diagnostics(cfg, trial_idx, ip, p_dbw, out, sol, urgent_idx, rate_vec_bps);
                norm_logged_this_p = true;
            end

            avg_qoe_all(trial_idx, ip, ia) = out.avg_qoe;
            urgent_qoe_all(trial_idx, ip, ia) = mean(out.qoe_vec(urgent_idx));
            normal_qoe_all(trial_idx, ip, ia) = mean(out.qoe_vec(normal_idx));
            sum_rate_all(trial_idx, ip, ia) = out.sum_rate_bps;
            urgent_sum_rate_all(trial_idx, ip, ia) = urgent_sum_rate_bps;
            urgent_avg_rate_all(trial_idx, ip, ia) = urgent_avg_rate_bps;
            urgent_delay_vio_all(trial_idx, ip, ia) = mean(out.delay_vio_vec(urgent_idx));
            urgent_semantic_vio_all(trial_idx, ip, ia) = mean(out.semantic_vio_vec(urgent_idx));
            Qd_mean_all(trial_idx, ip, ia) = out.Qd_mean_all;
            Qs_mean_all(trial_idx, ip, ia) = out.Qs_mean_all;
            ris_count_all(trial_idx, ip, ia) = sum(sol.assign(:) > 0);
        end
    end

    if mod(trial_idx, 5) == 0 || trial_idx == mc
        fprintf('  trial %d/%d\n', trial_idx, mc);
    end
end

stats = struct();
[stats.avg_qoe.mean, stats.avg_qoe.ci95] = calc_mean_ci(avg_qoe_all);
[stats.urgent_qoe.mean, stats.urgent_qoe.ci95] = calc_mean_ci(urgent_qoe_all);
[stats.normal_qoe.mean, stats.normal_qoe.ci95] = calc_mean_ci(normal_qoe_all);
[stats.sum_rate.mean, stats.sum_rate.ci95] = calc_mean_ci(sum_rate_all);
[stats.urgent_sum_rate.mean, stats.urgent_sum_rate.ci95] = calc_mean_ci(urgent_sum_rate_all);
[stats.urgent_avg_rate.mean, stats.urgent_avg_rate.ci95] = calc_mean_ci(urgent_avg_rate_all);
[stats.urgent_delay_vio.mean, stats.urgent_delay_vio.ci95] = calc_mean_ci(urgent_delay_vio_all);
[stats.urgent_semantic_vio.mean, stats.urgent_semantic_vio.ci95] = calc_mean_ci(urgent_semantic_vio_all);
[stats.Qd_mean.mean, stats.Qd_mean.ci95] = calc_mean_ci(Qd_mean_all);
[stats.Qs_mean.mean, stats.Qs_mean.ci95] = calc_mean_ci(Qs_mean_all);
[stats.ris_count.mean, stats.ris_count.ci95] = calc_mean_ci(ris_count_all);
[stats.proposed_eval_calls.mean, stats.proposed_eval_calls.ci95] = calc_mean_ci2d(proposed_eval_calls_all);
[stats.ga_rt_eval_calls.mean, stats.ga_rt_eval_calls.ci95] = calc_mean_ci2d(ga_rt_eval_calls_all);
[stats.ga_rt_budget_target.mean, stats.ga_rt_budget_target.ci95] = calc_mean_ci2d(ga_rt_budget_target_all);
stats = ensure_urgent_rate_metrics(stats);

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

run_id = out_name;
base_name = out_name;

if p.Results.save_figures
    [plot_idx, plot_labels] = get_plot_view(alg_names, p.Results.show_ga_ub);
    debug_print_power_rate_pairs(p_list, stats.urgent_sum_rate.mean(:, plot_idx) / 1e6, plot_labels, 5);
    plot_ci_figure(p_list, stats.urgent_qoe.mean(:, plot_idx), stats.urgent_qoe.ci95(:, plot_idx), plot_labels, ...
        'Urgent QoE Cost', 'Urgent QoE Cost vs p', fullfile(fig_dir, [base_name '_urgent_qoe.png']));
    plot_ci_figure(p_list, stats.normal_qoe.mean(:, plot_idx), stats.normal_qoe.ci95(:, plot_idx), plot_labels, ...
        'Normal QoE Cost', 'Normal QoE Cost vs p', fullfile(fig_dir, [base_name '_normal_qoe.png']));
    plot_ci_figure(p_list, stats.avg_qoe.mean(:, plot_idx), stats.avg_qoe.ci95(:, plot_idx), plot_labels, ...
        'Avg QoE Cost', 'Average QoE Cost vs p', fullfile(fig_dir, [base_name '_avg_qoe.png']));
    plot_ci_figure(p_list, stats.sum_rate.mean(:, plot_idx), stats.sum_rate.ci95(:, plot_idx), plot_labels, ...
        'Sum-Rate (bps)', 'Sum-Rate vs p', fullfile(fig_dir, [base_name '_sum_rate.png']));
    plot_ci_figure(p_list, stats.urgent_sum_rate.mean(:, plot_idx) / 1e6, stats.urgent_sum_rate.ci95(:, plot_idx) / 1e6, plot_labels, ...
        'Urgent Sum-Rate (Mbps)', 'Urgent Sum-Rate vs Power', fullfile(fig_dir, [base_name '_urgent_sum_rate.png']));
    plot_ci_figure(p_list, stats.urgent_delay_vio.mean(:, plot_idx), stats.urgent_delay_vio.ci95(:, plot_idx), plot_labels, ...
        'Urgent Delay Violation', 'Urgent Delay Violation vs p', fullfile(fig_dir, [base_name '_urgent_delay_vio.png']));
    plot_ci_figure(p_list, stats.urgent_semantic_vio.mean(:, plot_idx), stats.urgent_semantic_vio.ci95(:, plot_idx), plot_labels, ...
        'Urgent Semantic Violation', 'Urgent Semantic Violation vs p', fullfile(fig_dir, [base_name '_urgent_semantic_vio.png']));
end

mat_path = fullfile(res_dir, [base_name '.mat']);
json_path = fullfile(res_dir, [base_name '.json']);
csv_path = fullfile(res_dir, [base_name '.csv']);

if p.Results.save_mat
    save(mat_path, ...
        'avg_qoe_all', 'urgent_qoe_all', 'normal_qoe_all', 'sum_rate_all', ...
        'urgent_sum_rate_all', 'urgent_avg_rate_all', ...
        'urgent_delay_vio_all', 'urgent_semantic_vio_all', 'Qd_mean_all', 'Qs_mean_all', 'ris_count_all', ...
        'proposed_eval_calls_all', 'ga_rt_eval_calls_all', 'ga_rt_budget_target_all', ...
        'p_list', 'alg_names', 'stats', 'mc', 'seed');
end

json_obj = struct();
json_obj.run_id = base_name;
json_obj.mc = mc;
json_obj.seed = seed;
json_obj.p_list = p_list;
json_obj.alg_names = alg_names;
json_obj.metrics = stats;
json_obj.cfg_overrides = p.Results.cfg_overrides;
json_obj.ga_rt_cfg = struct( ...
    'pop_size', p.Results.ga_rt_pop_size, ...
    'num_generations', p.Results.ga_rt_num_generations, ...
    'budget_evals', p.Results.ga_rt_budget_evals, ...
    'light_solve_only', true);
json_obj.show_ga_ub = p.Results.show_ga_ub;
json_obj.include_ga_ub = p.Results.include_ga_ub;
json_obj.warm_start = logical(p.Results.warm_start);
json_obj.notes = 'GA-RT is real-time constrained GA baseline; GA-UB is upper-bound reference only.';
json_obj.sanity = sanity;
write_text_file(json_path, jsonencode(json_obj));

if p.Results.save_csv
    write_summary_csv(csv_path, p_list, alg_names, stats);
end

fprintf('\nSaved:\n');
fprintf('  %s\n', mat_path);
fprintf('  %s\n', json_path);
fprintf('  %s\n', csv_path);
fprintf('  figures: %s\n', fig_dir);
end

function [urgent_idx, normal_idx] = get_group_indices(cfg, profile)
K = cfg.num_users;
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('paper_sweep_power:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required for unified grouping.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
if ~isempty(intersect(urgent_idx, normal_idx))
    error('paper_sweep_power:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx(:); normal_idx(:)]);
assert(numel(cover) == K && all(cover(:) == (1:K).'), ...
    'paper_sweep_power:group_coverage: urgent/normal union must cover all users');
end

function [assign, info] = pick_assignment_local(cfg, ch, geom, mode, profile, p_dbw, proposed_budget, ga_log, run_opts, warm_assign)
K = cfg.num_users;
L = cfg.num_ris;
info = struct();

switch lower(mode)
    case 'random'
        % Capacity-feasible random baseline; direct link (0) is allowed by design.
        assign = pick_random_capacity(cfg);
    case 'norm'
        assign = pick_norm_capacity(cfg, ch);
    case 'ga_ub'
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'random_fixed';
        opts.pop_size = cfg.ga_Np;
        opts.num_generations = cfg.ga_Niter;
        opts.ga_log = ga_log;
        if ~isempty(warm_assign)
            opts.seed_assignment = warm_assign(:);
        end
        [assign, ~, ~, ga_info] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), L, cfg.k0);
        info = ga_info;
    case 'ga_rt'
        opts = struct();
        opts.geom = geom;
        opts.profile = profile;
        opts.semantic_mode = cfg.semantic_mode;
        opts.table_path = cfg.semantic_table;
        opts.theta_strategy = 'random_fixed';
        opts.pop_size = run_opts.ga_rt_pop_size;
        opts.num_generations = run_opts.ga_rt_num_generations;
        opts.ga_log = ga_log;
        target_budget = run_opts.ga_rt_budget_evals;
        if ~isempty(proposed_budget) && isfinite(proposed_budget) && proposed_budget > 0
            % Real-time GA can be aligned to measured proposed effort but must stay under GA-RT cap.
            target_budget = min(target_budget, proposed_budget);
        end
        opts.budget_evals = target_budget;
        if ~isempty(warm_assign)
            opts.seed_assignment = warm_assign(:);
        end
        info.budget_target = target_budget;
        [assign, ~, ~, ga_info] = ga_match_qoe(cfg, ch, p_dbw, opts);
        assign = enforce_capacity(assign(:), L, cfg.k0);
        info = ga_info;
        if ~isfield(info, 'budget_target')
            info.budget_target = target_budget;
        end
    otherwise
        assign = zeros(K, 1);
end
end

function assign = pick_random_capacity(cfg)
K = cfg.num_users;
L = cfg.num_ris;
cap = cfg.k0 * ones(L, 1);
assign = zeros(K, 1);
order = randperm(K);
for ii = 1:K
    k = order(ii);
    avail = find(cap > 0).';
    choices = [0, avail];
    pick = choices(randi(numel(choices)));
    if pick > 0
        cap(pick) = cap(pick) - 1;
    end
    assign(k) = pick;
end
end

function assign = pick_norm_capacity(cfg, ch)
K = cfg.num_users;
L = cfg.num_ris;
theta_ref = ch.theta;

pow_direct = zeros(K, 1);
pow_ris = zeros(K, L);
best_gain = zeros(K, 1);

for k = 1:K
    h0 = ch.h_d(:, k);
    pow_direct(k) = real(h0' * h0);
    for l = 1:L
        h_l = ch.h_d(:, k) + cfg.ris_gain * ch.G(:, :, l) * (theta_ref(:, l) .* ch.H_ris(:, k, l));
        pow_ris(k, l) = real(h_l' * h_l);
    end
    best_gain(k) = max(pow_ris(k, :)) - pow_direct(k);
end

cap = cfg.k0 * ones(L, 1);
assign = zeros(K, 1);
[~, order] = sort(best_gain, 'descend');
for ii = 1:K
    k = order(ii);
    if best_gain(k) <= 0
        assign(k) = 0;
        continue;
    end
    [~, ris_rank] = sort(pow_ris(k, :), 'descend');
    placed = false;
    for jj = 1:L
        l = ris_rank(jj);
        if cap(l) > 0 && pow_ris(k, l) > pow_direct(k)
            assign(k) = l;
            cap(l) = cap(l) - 1;
            placed = true;
            break;
        end
    end
    if ~placed
        assign(k) = 0;
    end
end
end

function assign = enforce_capacity(assign, L, k0)
assign = assign(:);
for l = 1:L
    idx = find(assign == l);
    if numel(idx) > k0
        drop_idx = idx((k0 + 1):end);
        assign(drop_idx) = 0;
    end
end
end

function sol = build_light_solution(cfg, ch, assign, p_dbw)
theta_all = ch.theta;
h_eff = effective_channel(cfg, ch, assign, theta_all);
[V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 3);
sol = struct('assign', assign(:), 'theta_all', theta_all, 'V', V);
end

function out = derive_algo_seed(base_seed, trial_idx, p_idx, alg_idx)
out = base_seed + trial_idx * 1000000 + p_idx * 10000 + alg_idx * 100;
end

function [plot_idx, plot_labels] = get_plot_view(alg_names, show_ga_ub)
plot_idx = 1:numel(alg_names);
if ~show_ga_ub
    ub = find(strcmpi(alg_names, 'ga_ub'));
    plot_idx(ismember(plot_idx, ub)) = [];
end
plot_labels = cell(1, numel(plot_idx));
for i = 1:numel(plot_idx)
    plot_labels{i} = legend_label(alg_names{plot_idx(i)});
end
end

function lbl = legend_label(name)
switch lower(name)
    case 'ga_rt'
        lbl = 'GA-RT';
    case 'ga_ub'
        lbl = 'GA-UB';
    otherwise
        lbl = name;
end
end

function [mu, ci95] = calc_mean_ci(x)
mc = size(x, 1);
num_p = size(x, 2);
num_alg = size(x, 3);
mu = reshape(mean(x, 1), [num_p, num_alg]);
sd = reshape(std(x, 0, 1), [num_p, num_alg]);
ci95 = 1.96 * sd / sqrt(mc);
end

function [mu, ci95] = calc_mean_ci2d(x)
mc = size(x, 1);
mu = mean(x, 1, 'omitnan');
sd = std(x, 0, 1, 'omitnan');
ci95 = 1.96 * sd / sqrt(mc);
mu = mu(:).';
ci95 = ci95(:).';
end

function plot_ci_figure(x, y_mean, y_ci, alg_names, y_label, fig_title, out_path)
A = numel(alg_names);
colors = lines(A);
markers = {'o', 's', 'd', '^', 'v', 'x'};

fig = figure('Color', 'w', 'Visible', 'off');
hold on;
for a = 1:A
    xa = x(:).';
    ya = y_mean(:, a).';
    plot(xa, ya, ['-' markers{a}], 'Color', colors(a, :), 'LineWidth', 1.6, ...
        'MarkerSize', 6, 'DisplayName', alg_names{a});
end
grid on;
xlabel('p (dBW)');
ylabel(y_label);
title(fig_title);
legend('Location', 'best');
saveas(fig, out_path);
close(fig);
end

function debug_print_power_rate_pairs(p_list, y_mbps, labels, max_points)
n_p = numel(p_list);
n_show = min(max_points, n_p);
fprintf('\n[debug] First %d (p_dBW, urgent_sum_rate_Mbps) pairs per method:\n', n_show);
for a = 1:numel(labels)
    fprintf('[debug] %s\n', labels{a});
    for i = 1:n_show
        fprintf('  p=%.6g dBW, rate=%.6g Mbps\n', p_list(i), y_mbps(i, a));
    end
end
end

function debug_print_norm_diagnostics(cfg, trial_idx, ip, p_dbw, out, sol, urgent_idx, rate_vec_bps)
gamma_norm = out.gamma_p(:);
rate_norm = rate_vec_bps(:);
gamma_c = out.gamma_c(:);
ck = sol.V.c(:);

has_nonfinite_gamma = any(~isfinite(gamma_norm));
has_nonfinite_rate = any(~isfinite(rate_norm));
gamma_min = min(gamma_norm);
gamma_max = max(gamma_norm);
one_plus_gamma = 1 + gamma_norm;
one_plus_gamma_min = min(one_plus_gamma);
one_plus_gamma_max = max(one_plus_gamma);
has_gamma_lt_minus1 = any(gamma_norm < -1);

tx_power = real(norm(sol.V.v_c)^2 + sum(sum(abs(sol.V.V_p).^2)));
is_tx_all_zero = tx_power <= 1e-12;
is_ck_all_zero = all(abs(ck) <= 1e-12);

rc_limit = cfg.bandwidth * log2(1 + min(gamma_c) + cfg.eps);
rc_alloc_sum = sum(ck);

urgent_sum_expr = 'urgent_sum_rate = sum(rate_vec_bps(urgent_idx))';
urgent_sum_dbg = sum(rate_norm(urgent_idx));
total_sum_dbg = sum(rate_norm);

fprintf('\n[norm-debug] trial=%d ip=%d p_dbw=%.6g\n', trial_idx, ip, p_dbw);
fprintf('[norm-debug] any(~isfinite(gamma_norm))=%d, any(~isfinite(rate_norm))=%d\n', ...
    has_nonfinite_gamma, has_nonfinite_rate);
fprintf('[norm-debug] min/max gamma_norm = [%.6g, %.6g]\n', gamma_min, gamma_max);
fprintf('[norm-debug] min/max (1+gamma_norm) = [%.6g, %.6g], any(gamma_norm<-1)=%d\n', ...
    one_plus_gamma_min, one_plus_gamma_max, has_gamma_lt_minus1);
fprintf('[norm-debug] tx_power(||vc||^2+sum||vk||^2)=%.6g, tx_all_zero=%d\n', tx_power, is_tx_all_zero);
fprintf('[norm-debug] ck_all_zero=%d, rc_limit=%.6g bps, sum(ck)=%.6g bps\n', ...
    is_ck_all_zero, rc_limit, rc_alloc_sum);
fprintf('[norm-debug] urgent_set_size=%d, urgent_set=%s\n', numel(urgent_idx), mat2str(urgent_idx(:).'));
fprintf('[norm-debug] expr: %s\n', urgent_sum_expr);
fprintf('[norm-debug] sum(rate_norm(urgent_set))=%.6g bps, total_sum_rate=%.6g bps, out.urgent_sum_rate_bps=%.6g bps\n', ...
    urgent_sum_dbg, total_sum_dbg, out.urgent_sum_rate_bps);
end

function debug_print_power_chain(cfg, trial_idx, ip, p_dbw, urgent_idx)
p_watts = 10^(p_dbw / 10);
fprintf('\n[chain-debug] trial=%d ip=%d p_dbw=%.6g p_watts=%.6g noise_watts=%.6g\n', ...
    trial_idx, ip, p_dbw, p_watts, cfg.noise_watts);
fprintf('[chain-debug] urgent_set_size=%d urgent_set=%s\n', numel(urgent_idx), mat2str(urgent_idx(:).'));
end

function sanity = update_sanity_bounds(sanity, out, tol_q, ctx)
q = out.qoe_vec(:);
qd = out.Qd_vec(:);
qs = out.Qs_vec(:);

if any(~isfinite(q)) || any(~isfinite(qd)) || any(~isfinite(qs))
    error('paper_sweep_power:nonfinite_qoe', 'Non-finite QoE/Qd/Qs at %s', ctx);
end
if any(q < -tol_q) || any(q > 1 + tol_q)
    error('paper_sweep_power:qoe_out_of_range', 'QoE out of [0,1] at %s', ctx);
end
if any(qd < -tol_q) || any(qd > 1 + tol_q)
    error('paper_sweep_power:qd_out_of_range', 'Qd out of [0,1] at %s', ctx);
end
if any(qs < -tol_q) || any(qs > 1 + tol_q)
    error('paper_sweep_power:qs_out_of_range', 'Qs out of [0,1] at %s', ctx);
end

sanity.qoe_min = min(sanity.qoe_min, min(q));
sanity.qoe_max = max(sanity.qoe_max, max(q));
sanity.Qd_min = min(sanity.Qd_min, min(qd));
sanity.Qd_max = max(sanity.Qd_max, max(qd));
sanity.Qs_min = min(sanity.Qs_min, min(qs));
sanity.Qs_max = max(sanity.Qs_max, max(qs));
end

function write_summary_csv(csv_path, p_list, alg_names, stats)
metrics = {'avg_qoe', 'urgent_qoe', 'normal_qoe', 'sum_rate', ...
           'urgent_sum_rate', 'urgent_avg_rate', ...
           'urgent_delay_vio', 'urgent_semantic_vio', 'Qd_mean', 'Qs_mean', 'ris_count'};
fid = fopen(csv_path, 'w');
if fid < 0
    error('Cannot open CSV file for writing: %s', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, 'p,alg,metric,mean,ci95\n');
for m = 1:numel(metrics)
    name = metrics{m};
    mu = stats.(name).mean;
    ci = stats.(name).ci95;
    for ip = 1:numel(p_list)
        for ia = 1:numel(alg_names)
            fprintf(fid, '%.10g,%s,%s,%.10g,%.10g\n', ...
                p_list(ip), alg_names{ia}, name, mu(ip, ia), ci(ip, ia));
        end
    end
end
clear cleanup_obj;
end

function write_text_file(path_name, txt)
fid = fopen(path_name, 'w');
if fid < 0
    error('Cannot open file for writing: %s', path_name);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', txt);
clear cleanup_obj;
end

function v = get_rate_vec_bps_compat(out, K)
if isfield(out, 'rate_vec_bps') && numel(out.rate_vec_bps) == K
    v = out.rate_vec_bps(:);
elseif isfield(out, 'rate_total_bps') && numel(out.rate_total_bps) == K
    v = out.rate_total_bps(:);
elseif isfield(out, 'sum_rate_bps') && isfinite(out.sum_rate_bps)
    % Compatibility fallback for legacy outputs without per-user rates.
    v = (out.sum_rate_bps / max(1, K)) * ones(K, 1);
else
    error('paper_sweep_power:missing_rate_vec', 'Cannot recover per-user rate vector.');
end
end

function idx = normalize_index_vector(x, K)
x = x(:);
if islogical(x)
    if numel(x) ~= K
        error('Index logical mask must be Kx1.');
    end
    idx = find(x);
    return;
end
if isnumeric(x) && numel(x) == K
    xr = round(real(x));
    if all((xr == 0) | (xr == 1))
        idx = find(xr > 0);
        return;
    end
end
idx = unique(round(real(x)));
idx = idx(isfinite(idx) & idx >= 1 & idx <= K);
idx = idx(:);
end

function stats = ensure_urgent_rate_metrics(stats)
if ~isfield(stats, 'sum_rate')
    return;
end
mu = stats.sum_rate.mean;
ci = stats.sum_rate.ci95;
if ~isfield(stats, 'urgent_sum_rate')
    stats.urgent_sum_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
if ~isfield(stats, 'urgent_avg_rate')
    stats.urgent_avg_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
end

function cfg = apply_cfg_overrides(cfg, overrides)
if isempty(overrides) || ~isstruct(overrides)
    return;
end
fn = fieldnames(overrides);
for i = 1:numel(fn)
    cfg.(fn{i}) = overrides.(fn{i});
end
end
