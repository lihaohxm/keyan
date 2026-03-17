function [bestChoice, bestQoE, histBest, ga_info] = ga_match_qoe(cfg, ch, p_dbw, opts)
%GA_MATCH_QOE Genetic Algorithm for RIS-user matching.
%
% Fairness notes:
% 1) theta strategy is consistent in fitness and final evaluation.
% 2) GA fitness can directly use external per-user profile (urgent/normal).
%
% theta_strategy:
%   'align_fixed'  (default): use per-assignment aligned theta consistently
%   'random_fixed' (legacy) : use ch.theta consistently

if nargin < 4, opts = struct(); end

% GA Parameters
Np = get_opt(opts, 'pop_size', get_opt(opts, 'Np', 20));
Niter = get_opt(opts, 'num_generations', get_opt(opts, 'Niter', 5));
Pc = get_opt(opts, 'Pc', 0.80);
Pm = get_opt(opts, 'Pm', 0.20);
elite = get_opt(opts, 'elite', 3);
inject = get_opt(opts, 'inject', 2);
t = get_opt(opts, 't', 3);
verbose = get_opt(opts, 'verbose', false);
enable_ga_log = get_opt(opts, 'ga_log', false) || verbose;
budget_evals = get_opt(opts, 'budget_evals', inf); % GA-budgeted support
max_time = get_opt(opts, 'max_time', inf);           % 时间截断支持

semantic_mode = get_opt(opts, 'semantic_mode', cfg.semantic_mode);
table_path = get_opt(opts, 'table_path', cfg.semantic_table);
geom = get_opt(opts, 'geom', []);
theta_strategy = lower(char(get_opt(opts, 'theta_strategy', 'align_fixed')));

if isfield(cfg, 'proxy_a') && isfield(cfg, 'proxy_b')
    proxy_params = struct('a', cfg.proxy_a, 'b', cfg.proxy_b);
else
    proxy_params = struct();
end

K = cfg.num_users;
L = cfg.num_ris;
K0 = cfg.k0;

profile_opt = get_opt(opts, 'profile', []);
weights_opt = get_opt(opts, 'weights', cfg.weights(1, :));
if ~isempty(profile_opt)
    profile_eval = profile_opt;
else
    % Legacy fallback (kept for compatibility with older callers).
    profile_eval = make_uniform_profile(cfg, weights_opt);
end

urgent_idx_eval = [];
if isfield(profile_eval, 'groups') && isfield(profile_eval.groups, 'urgent_idx')
    urgent_idx_eval = normalize_index_vector_local(profile_eval.groups.urgent_idx, K);
end
is_urgent_eval = false(K, 1);
is_urgent_eval(urgent_idx_eval) = true;
proxy_common_weights = build_proxy_common_weights(profile_eval, cfg);
alpha_c_proxy_default = get_opt(opts, 'alpha_c_proxy', get_cfg_local(cfg, 'common_init_ratio', 0.18));
alpha_c_proxy_default = min(max(alpha_c_proxy_default, 0.05), 0.40);
rc_scale_default = get_opt(opts, 'ga_proxy_rc_scale', 0.99);

num_hard = round(cfg.hard_ratio * K);
can_use_ris = false(K, 1);
can_use_ris(1:num_hard) = true;
ris_users = find(can_use_ris);
direct_only = find(~can_use_ris);

if isfinite(budget_evals) && budget_evals > 0
    % Keep the same mutation/crossover style but shrink GA scale to budget.
    Np = min(Np, max(4, floor(budget_evals / 2)));
end
elite = min(max(1, elite), max(1, Np - 2));
inject = min(max(0, inject), max(0, Np - elite - 1));
n_offspring = max(1, Np - elite - inject);

if isfinite(budget_evals) && budget_evals > 0
    eval_per_gen = max(1, Np - elite);
    Niter = min(Niter, max(0, floor((budget_evals - Np) / eval_per_gen)));
end

theta_base = ch.theta;
eval_opts = struct('semantic_mode', semantic_mode, 'table_path', table_path, 'semantic_params', proxy_params);
eval_count = 0;

t_ga_start = tic;
% ========== INITIALIZATION ==========
use_proposed_seed = get_opt(opts, 'use_proposed_seed', false);
seed_assignment = get_opt(opts, 'seed_assignment', []);
pop = zeros(K, Np);
for i = 1:Np
    pop(:, i) = random_assignment(K, L, K0, ris_users, direct_only);
end

if ~isempty(seed_assignment) && numel(seed_assignment) == K
    seed_assignment = seed_assignment(:);
    seed_assignment(direct_only) = 0;
    seed_assignment = repair(seed_assignment, L, K0);
    pop(:, 1) = seed_assignment;
end

if use_proposed_seed
    try
        proposed_assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, profile_eval.weights, geom, profile_eval);
        proposed_assign(direct_only) = 0;
        proposed_assign = repair(proposed_assign, L, K0);
        pop(:, 1) = proposed_assign;
    catch
    end
end

% ========== FITNESS ==========
fit = -1e9 * ones(1, Np);
bestFit = -1e9;
bestChoice = pop(:, 1);
bestOut = [];
for i = 1:Np
    if toc(t_ga_start) >= max_time % [Antigravity Fix]
        break;
    end
    [fit(i), out_i] = evaluate(pop(:, i));
    if fit(i) > bestFit
        bestFit = fit(i);
        bestChoice = pop(:, i);
        bestOut = out_i;
    end
end
histBest = zeros(1, Niter + 1);
histBest(1) = bestFit;

% ========== EVOLUTION ==========
for gen = 1:Niter
    % 只要达到评估次数，或者运行时间超标，立即中断进化
    if eval_count >= budget_evals || toc(t_ga_start) >= max_time
        break;
    end

    [sorted_fit, sortIdx] = sort(fit, 'descend');
    elitePop = pop(:, sortIdx(1:elite));
    eliteFit = sorted_fit(1:elite);

    mating_pool = zeros(K, Np);
    for i = 1:Np
        cands = randi(Np, 1, t);
        [~, w] = max(fit(cands));
        mating_pool(:, i) = pop(:, cands(w));
    end

    offspring = zeros(K, n_offspring);
    i = 1;
    while i <= n_offspring
        p1 = mating_pool(:, randi(Np));
        p2 = mating_pool(:, randi(Np));

        if rand < Pc
            pt = randi(K);
            c1 = [p1(1:pt); p2(pt + 1:end)];
            c2 = [p2(1:pt); p1(pt + 1:end)];
        else
            c1 = p1; c2 = p2;
        end

        c1 = flip_mutate(c1, L, ris_users, Pm);
        c2 = flip_mutate(c2, L, ris_users, Pm);
        c1(direct_only) = 0;
        c2(direct_only) = 0;
        c1 = repair(c1, L, K0);
        c2 = repair(c2, L, K0);

        offspring(:, i) = c1;
        if i + 1 <= n_offspring
            offspring(:, i + 1) = c2;
        end
        i = i + 2;
    end

    injected = zeros(K, inject);
    for i = 1:inject
        injected(:, i) = random_assignment(K, L, K0, ris_users, direct_only);
    end

    pop = [elitePop, offspring, injected];
    fit(1:elite) = eliteFit;
    for i = elite + 1:Np
        if toc(t_ga_start) >= max_time % [Antigravity Fix]
            break;
        end
        [fit(i), out_i] = evaluate(pop(:, i));
        if fit(i) > bestFit
            bestFit = fit(i);
            bestChoice = pop(:, i);
            bestOut = out_i;
        end
    end

    [gen_best, gen_best_idx] = max(fit);
    if gen_best > bestFit
        bestFit = gen_best;
        bestChoice = pop(:, gen_best_idx);
    end
    histBest(gen + 1) = bestFit;

    if enable_ga_log
        [~, out_best] = evaluate_raw(bestChoice);
        if ~isempty(out_best)
            fprintf('[GA][gen %d] fit=%.6f comp=%.6f qoe=%.6f delay_vio=%.4f sem_vio=%.4f eval=%d\n', ...
                gen, bestFit, get_eval_cost(out_best), out_best.avg_qoe, ...
                out_best.delay_vio_rate_all, out_best.semantic_vio_rate_all, eval_count);
        end
    end
end

% ========== FINAL REPORTING ==========
% Do not append an uncharged fixed-X refinement here. This helper should
% report the GA search result itself so RT/budgeted usage remains fair.
out_final = bestOut;

if isempty(out_final)
    bestQoE = inf;
else
    bestQoE = get_eval_cost(out_final);
end

if enable_ga_log && ~isempty(out_final)
    fprintf('[GA][final] proxy_fit=%.6f comp=%.6f proxy_qoe=%.6f delay_vio=%.4f sem_vio=%.4f eval=%d budget=%g\n', ...
        bestFit, get_eval_cost(out_final), out_final.avg_qoe, ...
        out_final.delay_vio_rate_all, out_final.semantic_vio_rate_all, eval_count, budget_evals);
end

ga_info = struct();
ga_info.eval_count = eval_count;
ga_info.theta_strategy = theta_strategy;
ga_info.budget_evals = budget_evals;
ga_info.Np = Np;
ga_info.Niter = Niter;
ga_info.pop_size = Np;
ga_info.num_generations = Niter;
ga_info.best_fit = bestFit;
ga_info.mode = ternary(isfinite(budget_evals), 'ga_rt', 'ga_ub');

    function [fit_val, out] = evaluate(choice)
        if eval_count >= budget_evals
            fit_val = -1e9;
            out = [];
            return;
        end
        [fit_val, out] = evaluate_raw(choice);
        eval_count = eval_count + 1;
    end

    function [fit_val, out] = evaluate_raw(choice)
        if ~is_feasible(choice, L, K0)
            fit_val = -1e9;
            out = [];
            return;
        end
        % --- 阶段一：快速代理评估 (RZF + 保守功率分配) ---
        theta_eval = resolve_theta(theta_strategy, cfg, ch, choice, theta_base);
        K_u = cfg.num_users;
        nt = size(ch.h_d, 1);
        p_watts = 10^(p_dbw / 10);
        noise_power = cfg.noise_watts;
        h_eff = zeros(nt, K_u);
        for k = 1:K_u
            hk = ch.h_d(:, k);
            l = choice(k);
            if l > 0 && l <= L
                hk = hk + cfg.ris_gain * ch.G(:, :, l) * (theta_eval(:, l) .* ch.H_ris(:, k, l));
            end
            h_eff(:, k) = hk;
        end

        % ★ 修复 A：将公共流功率比例从 20% 降至 10%，与 WMMSE 初始化对齐
        % 原先 20% 导致代理模型的公共流 SINR 系统性高估，误导进化方向
        alpha_c_proxy = alpha_c_proxy_default;
        v_c_proxy = sum(h_eff, 2);
        v_c_proxy = v_c_proxy / (norm(v_c_proxy) + cfg.eps) * sqrt(p_watts * alpha_c_proxy);
        p_p_watts = p_watts * (1 - alpha_c_proxy);

        HH = h_eff' * h_eff;
        reg_val = max(noise_power, real(trace(HH)) / (K_u * 100)) + 1e-8;
        V_p_zf = h_eff / (HH + reg_val * eye(K_u));
        V_p = zeros(nt, K_u);
        for k = 1:K_u
            V_p(:, k) = V_p_zf(:, k) / (norm(V_p_zf(:, k)) + cfg.eps) * sqrt(p_p_watts / K_u);
        end

        % ★ 修复 B：使用"SIC 尚未执行"的保守 SINR 分母计算 gamma_c_proxy
        % 原先分母不含 v_c_proxy 自身功率（ZF 对消后），导致公共 SINR 虚高
        gamma_c_proxy = zeros(K_u, 1);
        for k = 1:K_u
            hk = h_eff(:, k);
            % 保守分母：所有私有流干扰 + 公共流自身功率 + 噪声
            denom_c = sum(abs(hk' * V_p).^2) + abs(hk' * v_c_proxy)^2 + noise_power;
            gamma_c_proxy(k) = abs(hk' * v_c_proxy)^2 / (denom_c + cfg.eps);
        end

        % ★ 修复 B：对 R_c_limit 施加 0.5 折扣（原 0.99），
        % 补偿代理模型与真实 WMMSE 之间的系统性高估偏差，
        % 确保 GA 进化方向不会被"虚高"的公共速率收益误导
        R_c_limit = rc_scale_default * cfg.bandwidth * log2(1 + min(gamma_c_proxy));

        % 按用户维度合并权重，确保 c_proxy 是 K_u x 1 向量
        c_proxy = allocate_common_rate_proxy(R_c_limit, proxy_common_weights, is_urgent_eval, cfg);
        common_marginal_gain_proxy = norm(sum(h_eff, 2))^2 / ...
            max(K_u * norm(h_eff, 'fro')^2, cfg.eps);
        common_cap_target = resolve_common_cap_proxy(cfg, profile_eval, common_marginal_gain_proxy);
        Pc_proxy = norm(v_c_proxy)^2;
        Pp_proxy = sum(sum(abs(V_p).^2));
        Pt_proxy = max(Pc_proxy + Pp_proxy, 1e-12);
        raw_common_ratio = Pc_proxy / Pt_proxy;
        common_excess = max(0, raw_common_ratio - common_cap_target);

        % 构造完整的 RSMA 代理结构
        V_proxy = struct('v_c', v_c_proxy, 'V_p', V_p, 'c', c_proxy);
        V_proxy.diag = struct( ...
            'common_power_ratio_raw', raw_common_ratio, ...
            'common_power_ratio_clipped', raw_common_ratio, ...
            'common_power_ratio', raw_common_ratio, ...
            'common_cap_target', common_cap_target, ...
            'common_cap_active', double(raw_common_ratio >= common_cap_target - 1e-9), ...
            'common_excess_penalty', get_cfg_local(cfg, 'common_excess_penalty_weight', 1.0) * common_excess.^2, ...
            'private_first_budget_ratio', Pp_proxy / Pt_proxy, ...
            'Rc_limit', R_c_limit, ...
            'private_rate_sum', nan, ...
            'common_rate_sum', sum(c_proxy));
        sol_eval = struct('assign', choice, 'theta_all', theta_eval, 'V', V_proxy);

        out = evaluate_system_rsma(cfg, ch, geom, sol_eval, profile_eval, eval_opts);
        fit_val = -get_eval_cost(out);
    end
end

function theta_eval = resolve_theta(theta_strategy, cfg, ch, choice, theta_base)
switch lower(theta_strategy)
    case 'align_fixed'
        theta_eval = build_align_theta(cfg, ch, choice, theta_base);
    otherwise
        theta_eval = theta_base;
end
end

function theta_all = build_align_theta(cfg, ch, assign, theta_base)
theta_all = theta_base;
L = cfg.num_ris;
for l = 1:L
    users_l = find(assign == l);
    if isempty(users_l)
        continue;
    end
    k = users_l(1);
    hdk = ch.h_d(:, k);
    w0 = hdk / (norm(hdk) + cfg.eps);
    proj = (w0' * ch.G(:, :, l)).' .* ch.H_ris(:, k, l);
    theta_all(:, l) = exp(-1j * angle(proj + cfg.eps));
end
end

function v = get_opt(opts, f, d)
if isfield(opts, f) && ~isempty(opts.(f))
    v = opts.(f);
else
    v = d;
end
end

function v = get_cfg_local(cfg, f, d)
if isfield(cfg, f) && ~isempty(cfg.(f))
    v = cfg.(f);
else
    v = d;
end
end

function cost = get_eval_cost(out)
if isempty(out)
    cost = inf;
elseif isfield(out, 'composite_cost') && ~isempty(out.composite_cost) && isfinite(out.composite_cost)
    cost = out.composite_cost;
else
    cost = out.avg_qoe;
end
end

function idx = normalize_index_vector_local(x, K)
if isempty(x)
    idx = zeros(0, 1);
    return;
end
if islogical(x)
    idx = find(x(:));
else
    idx = round(x(:));
    idx = idx(isfinite(idx));
    idx = idx(idx >= 1 & idx <= K);
end
idx = unique(idx, 'stable');
end

function y = normalize_vec_local(x)
x = real(x(:));
x(~isfinite(x)) = 0;
x = max(x, 0);
s = sum(x);
if s <= 0
    y = ones(size(x)) / max(numel(x), 1);
else
    y = x / s;
end
end

function w = build_proxy_common_weights(profile, cfg)
K = size(profile.weights, 1);
weight_sum = max(sum(profile.weights, 2), cfg.eps);
delay_pref = profile.weights(:, 1) ./ weight_sum;
semantic_pref = profile.weights(:, min(2, size(profile.weights, 2))) ./ weight_sum;

priority_score = ones(K, 1);
if isfield(profile, 'priority_score') && ~isempty(profile.priority_score)
    priority_score = profile.priority_score(:);
end

delay_pressure = 1 ./ max(profile.d_k(:), cfg.eps);
semantic_pressure = 1 ./ max(profile.dmax_k(:), cfg.eps);

w = 0.45 * normalize_vec_local(priority_score) + ...
    0.20 * normalize_vec_local(delay_pref) + ...
    0.20 * normalize_vec_local(semantic_pref) + ...
    0.15 * normalize_vec_local(0.5 * delay_pressure + 0.5 * semantic_pressure);
w = normalize_vec_local(w);
end

function c = allocate_common_rate_proxy(R_c_limit, weights, is_urgent, cfg)
K = numel(weights);
c = zeros(K, 1);
if R_c_limit <= 0
    return;
end

urgent_idx = find(is_urgent(:) > 0);
if isempty(urgent_idx)
    ww = sqrt(max(weights(:), 1e-12));
    c = R_c_limit * ww / sum(ww);
    return;
end

c_floor = get_cfg_local(cfg, 'common_rate_floor_urgent', 0.003e6);
total_floor = min(c_floor * numel(urgent_idx), ...
    get_cfg_local(cfg, 'common_rate_floor_share', 0.04) * R_c_limit);

if total_floor >= R_c_limit
    c(urgent_idx) = R_c_limit / numel(urgent_idx);
    return;
end

c(urgent_idx) = c_floor;
remain = R_c_limit - total_floor;
alloc_weights = sqrt(max(weights(:), 1e-12));
alloc_weights(urgent_idx) = alloc_weights(urgent_idx) .* ...
    (1 + get_cfg_local(cfg, 'common_rate_urgent_bonus', 0.10));
c = c + remain * alloc_weights / sum(alloc_weights);
end

function common_cap = resolve_common_cap_proxy(cfg, profile, common_gain)
cap_base = get_cfg_local(cfg, 'common_power_cap_base', get_cfg_local(cfg, 'common_power_cap_max', 0.38));
cap_min = get_cfg_local(cfg, 'common_power_cap_min', min(cap_base, 0.24));
cap_upper = get_cfg_local(cfg, 'common_power_cap_upper', max(cap_base, 0.42));
gain_term = get_cfg_local(cfg, 'common_cap_gain_weight', 0.16) * common_gain;

K = size(profile.weights, 1);
urgent_idx = [];
if isfield(profile, 'groups') && isfield(profile.groups, 'urgent_idx')
    urgent_idx = normalize_index_vector_local(profile.groups.urgent_idx, K);
end
if isempty(urgent_idx)
    urgent_idx = (1:K).';
end

weight_sum = max(sum(profile.weights, 2), cfg.eps);
delay_pref = profile.weights(:, 1) ./ weight_sum;
semantic_pref = profile.weights(:, min(2, size(profile.weights, 2))) ./ weight_sum;
delay_pressure = normalize_vec_local(1 ./ max(profile.d_k(:), cfg.eps));
semantic_pressure = normalize_vec_local(1 ./ max(profile.dmax_k(:), cfg.eps));

private_bias = ...
    get_cfg_local(cfg, 'common_cap_semantic_weight', 0.05) * ...
        mean(semantic_pref(urgent_idx) .* semantic_pressure(urgent_idx)) + ...
    get_cfg_local(cfg, 'common_cap_delay_weight', 0.03) * ...
        mean(delay_pref(urgent_idx) .* delay_pressure(urgent_idx));

common_cap = min(cap_upper, max(cap_min, cap_base + gain_term - private_bias));
if isfield(cfg, 'common_power_cap_max') && ~isempty(cfg.common_power_cap_max)
    common_cap = min(common_cap, cfg.common_power_cap_max);
end
end

function choice = random_assignment(K, L, K0, ris_users, direct_only)
choice = zeros(K, 1);
cap = K0 * ones(L, 1);
order = randperm(numel(ris_users));
for i = 1:numel(order)
    k = ris_users(order(i));
    avail = find(cap > 0);
    opts = [0; avail(:)];
    pick = opts(randi(numel(opts)));
    choice(k) = pick;
    if pick > 0
        cap(pick) = cap(pick) - 1;
    end
end
choice(direct_only) = 0;
end

function ok = is_feasible(choice, L, K0)
ok = true;
for l = 1:L
    if sum(choice == l) > K0
        ok = false;
        return;
    end
end
end

function choice = repair(choice, L, K0)
for l = 1:L
    users = find(choice == l);
    excess = numel(users) - K0;
    if excess > 0
        victims = users(randperm(numel(users), excess));
        choice(victims) = 0;
    end
end
end

function child = flip_mutate(child, L, ris_users, Pm)
for i = 1:numel(ris_users)
    k = ris_users(i);
    if rand < Pm
        if child(k) == 0
            child(k) = randi([1, L]);
        else
            if rand < 0.5
                child(k) = 0;
            else
                child(k) = randi([1, L]);
            end
        end
    end
end
end

function profile = make_uniform_profile(cfg, weights)
K = cfg.num_users;
profile = struct();
profile.M_k = cfg.m_k * ones(K, 1);
profile.weights = repmat(weights(:).', K, 1);
profile.d_k = default_deadlines(cfg, K);
profile.dmax_k = cfg.dmax * ones(K, 1);
end

function d_vec = default_deadlines(cfg, K)
num_hard = round(cfg.hard_ratio * K);
hard_mask = false(K, 1);
hard_mask(1:num_hard) = true;
if numel(cfg.deadlines) == 2
    d_vec = cfg.deadlines(1) * ones(K, 1);
    d_vec(~hard_mask) = cfg.deadlines(2);
else
    d_vec = cfg.deadlines(1) * ones(K, 1);
end
end

function out = ternary(cond, a, b)
if cond
    out = a;
else
    out = b;
end
end
