function run_id = paper_cdf(varargin)
%PAPER_CDF Urgent-user CDF plots at a fixed power point.
% Usage:
%   paper_cdf('mc',50,'p_dbw',-20,'seed',42)
%
% RNG fairness protocol:
% 1) Set rng(seed + trial_idx) once at each MC trial start.
% 2) Do not call rng/reset inside p/algorithm loops of that trial.
% 3) Algorithm-side randomness must use current RNG state.
% 4) geom/ch/profile are generated once per trial and reused by all algs.

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg = config();

p = inputParser;
addParameter(p, 'mc', 50);
addParameter(p, 'p_dbw', -20);
addParameter(p, 'seed', 42);
parse(p, varargin{:});

mc = p.Results.mc;
p_dbw = p.Results.p_dbw;
seed = p.Results.seed;

alg_names = {'proposed', 'random', 'norm', 'ga'};
num_alg = numel(alg_names);

qoe_samples_by_alg = cell(num_alg, 1);
dr_samples_by_alg = cell(num_alg, 1);
sr_samples_by_alg = cell(num_alg, 1);
for a = 1:num_alg
    qoe_samples_by_alg{a} = [];
    dr_samples_by_alg{a} = [];
    sr_samples_by_alg{a} = [];
end

opts_eval = struct('semantic_mode', cfg.semantic_mode, 'table_path', cfg.semantic_table);

fprintf('========================================\n');
fprintf('paper_cdf: MC=%d, p_dbw=%.2f, seed=%d\n', mc, p_dbw, seed);
fprintf('algorithms: %s\n', strjoin(alg_names, ', '));
fprintf('========================================\n');

for trial_idx = 1:mc
    trial_seed = seed + trial_idx;
    rng(trial_seed, 'twister');

    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);
    profile = build_profile_urgent_normal(cfg, geom, struct());
    urgent_idx = get_urgent_indices(profile, cfg.num_users);

    for a = 1:num_alg
        alg = alg_names{a};
        if strcmpi(alg, 'proposed')
            [assign, theta_all, V, ~] = ua_qoe_ao(cfg, ch, geom, p_dbw, cfg.semantic_mode, cfg.semantic_table, profile);
            sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
        else
            assign_fixed = pick_assignment_local(cfg, ch, geom, alg, profile, p_dbw);
            [sol_fixed, ~] = ua_qoe_ao_fixedX(cfg, ch, geom, p_dbw, assign_fixed, profile, struct());
            sol = sol_fixed;
        end

        out = evaluate_system_rsma(cfg, ch, geom, sol, profile, opts_eval);

        qoe_samples_by_alg{a} = [qoe_samples_by_alg{a}; out.qoe_vec(urgent_idx)];
        dr_samples_by_alg{a} = [dr_samples_by_alg{a}; out.T_tx(urgent_idx) ./ profile.d_k(urgent_idx)];
        sr_samples_by_alg{a} = [sr_samples_by_alg{a}; out.D(urgent_idx) ./ profile.dmax_k(urgent_idx)];
    end

    if mod(trial_idx, 5) == 0 || trial_idx == mc
        fprintf('  trial %d/%d\n', trial_idx, mc);
    end
end

function urgent_idx = get_urgent_indices(profile, K)
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx')
    error('paper_cdf:missing_groups', 'profile.groups.urgent_idx is required.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
if isempty(urgent_idx)
    error('paper_cdf:empty_urgent', 'profile.groups.urgent_idx resolved to empty.');
end
end

fig_dir = fullfile(proj_root, 'figures');
res_dir = fullfile(proj_root, 'results');
if ~exist(fig_dir, 'dir'), mkdir(fig_dir); end
if ~exist(res_dir, 'dir'), mkdir(res_dir); end

plot_cdf_set(qoe_samples_by_alg, alg_names, 'QoE Cost', 'Urgent QoE Cost CDF', fullfile(fig_dir, 'cdf_urgent_qoe.png'));
plot_cdf_set(dr_samples_by_alg, alg_names, 'Delay Ratio (T/d)', 'Urgent Delay Ratio CDF', fullfile(fig_dir, 'cdf_urgent_delay_ratio.png'));
plot_cdf_set(sr_samples_by_alg, alg_names, 'Semantic Ratio (D/dmax)', 'Urgent Semantic Ratio CDF', fullfile(fig_dir, 'cdf_urgent_sem_ratio.png'));

run_id = datestr(now, 'yyyymmdd_HHMMSS');
mat_path = fullfile(res_dir, ['cdf_urgent_' run_id '.mat']);
save(mat_path, 'alg_names', 'qoe_samples_by_alg', 'dr_samples_by_alg', 'sr_samples_by_alg', ...
    'mc', 'p_dbw', 'seed');

fprintf('\nSaved:\n');
fprintf('  %s\n', fullfile(fig_dir, 'cdf_urgent_qoe.png'));
fprintf('  %s\n', fullfile(fig_dir, 'cdf_urgent_delay_ratio.png'));
fprintf('  %s\n', fullfile(fig_dir, 'cdf_urgent_sem_ratio.png'));
fprintf('  %s\n', mat_path);
end

function plot_cdf_set(samples_by_alg, alg_names, x_label, fig_title, out_path)
colors = lines(numel(alg_names));
markers = {'o', 's', 'd', '^'};
fig = figure('Color', 'w', 'Visible', 'off');
hold on;
for a = 1:numel(alg_names)
    [xs, ys] = my_ecdf(samples_by_alg{a});
    if isempty(xs)
        continue;
    end
    plot(xs, ys, ['-' markers{a}], 'Color', colors(a, :), 'LineWidth', 1.6, ...
        'MarkerSize', 4, 'DisplayName', alg_names{a});
end
grid on;
xlabel(x_label);
ylabel('CDF');
title(fig_title);
ylim([0 1]);
legend('Location', 'best');
saveas(fig, out_path);
close(fig);
end

function [xs, ys] = my_ecdf(x)
x = x(~isnan(x) & ~isinf(x));
xs = sort(x(:));
n = numel(xs);
ys = (1:n).' / max(n, 1);
end

function assign = pick_assignment_local(cfg, ch, geom, mode, profile, p_dbw)
K = cfg.num_users;
L = cfg.num_ris;

switch lower(mode)
    case 'random'
        assign = pick_random_capacity(cfg);
    case 'norm'
        assign = pick_norm_capacity(cfg, ch);
    case 'ga'
        if exist('ga_match_qoe', 'file') == 2
            opts = struct();
            opts.geom = geom;
            opts.semantic_mode = cfg.semantic_mode;
            opts.table_path = cfg.semantic_table;
            opts.weights = mean(profile.weights, 1);
            opts.Np = cfg.ga_Np;
            opts.Niter = cfg.ga_Niter;
            [assign, ~, ~] = ga_match_qoe(cfg, ch, p_dbw, opts);
            assign = enforce_capacity(assign(:), L, cfg.k0);
        else
            % If GA matcher is unavailable, degrade to norm-based assignment.
            assign = pick_norm_capacity(cfg, ch);
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
