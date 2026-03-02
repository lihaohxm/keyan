function out = evaluate_system_rsma(cfg, ch, geom, sol, profile, opts)
%EVALUATE_SYSTEM_RSMA Unified RSMA evaluation entry.

if nargin < 6 || isempty(opts), opts = struct(); end
if nargin < 5, profile = []; end

K = cfg.num_users;
noise = cfg.noise_watts;

assign = sol.assign(:);
if numel(assign) ~= K
    error('sol.assign must be Kx1.');
end

if isfield(sol, 'theta_all') && ~isempty(sol.theta_all)
    theta_all = sol.theta_all;
else
    theta_all = ch.theta;
end

V = sol.V;
if ~isfield(V, 'v_c') || ~isfield(V, 'V_p') || ~isfield(V, 'c')
    error('sol.V must contain v_c, V_p, c.');
end
v_c = V.v_c;
V_p = V.V_p;
c_vec = V.c(:);

if isempty(profile)
    profile = build_profile_urgent_normal(cfg, geom, struct());
else
    profile = normalize_profile(profile, K);
end

semantic_mode = get_opt(opts, 'semantic_mode', cfg.semantic_mode);
table_path = get_opt(opts, 'table_path', cfg.semantic_table);
if isfield(opts, 'semantic_params') && ~isempty(opts.semantic_params)
    semantic_params = opts.semantic_params;
elseif isfield(cfg, 'proxy_a') && isfield(cfg, 'proxy_b')
    semantic_params = struct('a', cfg.proxy_a, 'b', cfg.proxy_b);
else
    semantic_params = struct();
end
prop_delay_opts = get_opt(opts, 'prop_delay_opts', struct());
debug_eval = get_opt(opts, 'debug_eval', false);

gamma_c = zeros(K, 1);
gamma_p = zeros(K, 1);

for k = 1:K
    hk = ch.h_d(:, k);
    l = assign(k);
    if l > 0
        hk = hk + cfg.ris_gain * ch.G(:, :, l) * (theta_all(:, l) .* ch.H_ris(:, k, l));
    end

    interf_all = sum(abs(hk' * V_p).^2);
    gamma_c(k) = abs(hk' * v_c)^2 / (interf_all + noise);

    interf_private = interf_all - abs(hk' * V_p(:, k))^2;
    gamma_p(k) = abs(hk' * V_p(:, k))^2 / (interf_private + noise);
end

rate_private_bps = cfg.bandwidth .* log2(1 + gamma_p + cfg.eps);
% Reuse existing RSMA definition: per-user total rate = private rate + allocated common-rate share.
rate_total_bps = (rate_private_bps + c_vec);
rate_total_bps = rate_total_bps(:); % Kx1, unit: bps
sum_rate_bps = sum(rate_total_bps);
group_idx = get_group_indices_from_profile(profile, K);
urgent_idx = group_idx.urgent_idx;
normal_idx = group_idx.normal_idx;
urgent_sum_rate_bps = sum(rate_total_bps(urgent_idx)); % bps
urgent_avg_rate_bps = mean(rate_total_bps(urgent_idx)); % bps

% Semantic mapping must use gamma_p (private stream SINR).
xi = semantic_map(gamma_p, profile.M_k, semantic_mode, table_path, semantic_params);
D = 1 - xi;

prop_delay = calc_prop_delay(cfg, geom, assign, prop_delay_opts);
[avg_qoe, qoe_vec, meta] = qoe( ...
    cfg, gamma_p, profile.M_k, xi, profile.weights, prop_delay, ...
    rate_total_bps, profile.d_k, profile.dmax_k);

delay_vio_vec = (meta.T_tx > profile.d_k);
semantic_vio_vec = (D > profile.dmax_k);

out.sum_rate_bps = sum_rate_bps;
out.rate_vec_bps = rate_total_bps; % Kx1 per-user total rate, bps
out.urgent_sum_rate_bps = urgent_sum_rate_bps; % bps
out.urgent_avg_rate_bps = urgent_avg_rate_bps; % bps
out.avg_qoe = avg_qoe;
out.qoe_vec = qoe_vec;
out.gamma_p = gamma_p;
out.gamma_c = gamma_c;
out.rate_private_bps = rate_private_bps;
out.rate_total_bps = rate_total_bps;
out.xi = xi;
out.D = D;
out.T_tx = meta.T_tx;
out.delay_vio_vec = logical(delay_vio_vec);
out.semantic_vio_vec = logical(semantic_vio_vec);
out.delay_vio_rate_all = mean(delay_vio_vec);
out.semantic_vio_rate_all = mean(semantic_vio_vec);
out.Qd_mean_all = mean(meta.Qd);
out.Qs_mean_all = mean(meta.Qs);
out.Qd_vec = meta.Qd(:);
out.Qs_vec = meta.Qs(:);

if debug_eval
    rate_gap = abs(sum(out.rate_vec_bps) - out.sum_rate_bps) / max(1, out.sum_rate_bps);
    if rate_gap >= 1e-6
        error('evaluate_system_rsma:rate_consistency_fail', ...
            'sum(rate_vec_bps) mismatch: rel_gap=%.3e', rate_gap);
    end
    gq = quantile(real(gamma_p), [0.1 0.5 0.9]);
    xq = quantile(real(xi), [0.1 0.5 0.9]);
    out.debug = struct();
    out.debug.gamma_p_p10 = gq(1);
    out.debug.gamma_p_p50 = gq(2);
    out.debug.gamma_p_p90 = gq(3);
    out.debug.xi_p10 = xq(1);
    out.debug.xi_p50 = xq(2);
    out.debug.xi_p90 = xq(3);
    out.debug.semantic_vio_rate = out.semantic_vio_rate_all;
    out.debug.rate_consistency_rel_gap = rate_gap;
    out.debug.urgent_count = numel(urgent_idx);
    out.debug.normal_count = numel(normal_idx);
    fprintf('[eval] gamma_p(P10/P50/P90)=(%.4g/%.4g/%.4g) xi(P10/P50/P90)=(%.4g/%.4g/%.4g) sem_vio=%.4f\n', ...
        gq(1), gq(2), gq(3), xq(1), xq(2), xq(3), out.semantic_vio_rate_all);
    head_n = min(6, numel(urgent_idx));
    fprintf('[eval] urgent_idx_count=%d urgent_head=%s\n', numel(urgent_idx), mat2str(urgent_idx(1:head_n).'));
end
end

function profile = default_profile(cfg, K)
    profile = struct();
    profile.M_k = cfg.m_k * ones(K, 1);
    profile.weights = repmat(cfg.weights(1, :), K, 1);
    profile.d_k = default_deadlines(cfg, K);
    profile.dmax_k = cfg.dmax * ones(K, 1);
    num_urgent = min(K, max(0, round(get_opt(cfg, 'num_urgent', K))));
    profile.groups.urgent_idx = (1:num_urgent).';
    profile.groups.normal_idx = ((num_urgent + 1):K).';
end

function profile = normalize_profile(profile, K)
    if ~isfield(profile, 'M_k') || isempty(profile.M_k)
        error('profile.M_k is required.');
    end
    if ~isfield(profile, 'weights') || isempty(profile.weights)
        error('profile.weights is required.');
    end
    if ~isfield(profile, 'd_k') || isempty(profile.d_k)
        error('profile.d_k is required.');
    end
    if ~isfield(profile, 'dmax_k') || isempty(profile.dmax_k)
        error('profile.dmax_k is required.');
    end

    profile.M_k = to_kvec(profile.M_k, K, 'profile.M_k');
    if size(profile.weights, 1) == 1 && size(profile.weights, 2) == 2
        profile.weights = repmat(profile.weights, K, 1);
    end
    if ~(size(profile.weights, 1) == K && size(profile.weights, 2) == 2)
        error('profile.weights must be Kx2 or 1x2.');
    end
    profile.d_k = to_kvec(profile.d_k, K, 'profile.d_k');
    profile.dmax_k = to_kvec(profile.dmax_k, K, 'profile.dmax_k');
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

function v = to_kvec(x, K, name)
    if numel(x) == 1
        v = repmat(x, K, 1);
    elseif numel(x) == K
        v = x(:);
    else
        error('%s must be scalar or Kx1.', name);
    end
end

function v = get_opt(s, f, d)
    if isfield(s, f) && ~isempty(s.(f))
        v = s.(f);
    else
        v = d;
    end
end

function out = get_group_indices_from_profile(profile, K)
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx') || ~isfield(profile.groups, 'normal_idx')
    error('evaluate_system_rsma:missing_groups', ...
        'profile.groups.{urgent_idx,normal_idx} are required as the single group source.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
normal_idx = normalize_index_vector(profile.groups.normal_idx, K);
urgent_idx = unique(urgent_idx(:));
normal_idx = unique(normal_idx(:));
if ~isempty(intersect(urgent_idx, normal_idx))
    error('evaluate_system_rsma:group_overlap', 'urgent_idx and normal_idx overlap.');
end
cover = unique([urgent_idx; normal_idx]);
if numel(cover) ~= K || any(cover(:) ~= (1:K).')
    error('evaluate_system_rsma:group_coverage', 'urgent_idx U normal_idx must cover all users exactly once.');
end
out = struct('urgent_idx', urgent_idx, 'normal_idx', normal_idx);
end
