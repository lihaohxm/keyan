function assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights, geom, profile, sol_ref)
%QOE_AWARE QoE-aware RIS-UE assignment via QoE-gain matching.
%
% Compatible inputs:
%   weights: 1x2 or Kx2
%   profile (optional struct): use profile.weights/profile.d_k/profile.dmax_k/profile.M_k
%   sol_ref (optional struct): if provided (with fields V, theta_all, assign),
%       compute QoE gains using the *same* RSMA+RIS configuration via
%       evaluate_system_rsma (paper-consistent gain table).

if nargin < 6 || isempty(weights)
    weights = cfg.weights(1, :);
end
if nargin < 7
    geom = [];
end
if nargin < 8
    profile = [];
end
if nargin < 9
    sol_ref = [];
end

K = cfg.num_users;
L = cfg.num_ris;
k0 = cfg.k0;

p_watts = 10.^(p_dbw / 10);
p_per_user = p_watts / K;

profile = build_local_profile(cfg, weights, profile, geom);
sem_params = local_semantic_params(cfg);
vio_bias_d = get_field_default(cfg, 'ao_vio_bias_d', 0);
vio_bias_s = get_field_default(cfg, 'ao_vio_bias_s', 0);
urgent_mult = get_field_default(cfg, 'ao_urgent_bias_mult', 1);
urgent_mask = false(K, 1);
if ~isfield(profile, 'groups') || ~isfield(profile.groups, 'urgent_idx')
    error('qoe_aware:missing_urgent_group', ...
        'profile.groups.urgent_idx is required as the single group source.');
end
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
urgent_mask(urgent_idx) = true;

% ------------------------------
% Full-evaluation gain table (paper-consistent):
% Keep (V, theta_all) fixed and compute per-user QoE costs for
%   - direct: assign(k)=0
%   - RIS l:  assign(k)=l
% with other users held at sol_ref.assign.
% ------------------------------
use_full = true;
if isfield(cfg, 'ao_match_full') && ~cfg.ao_match_full
    use_full = false;
end

if use_full && ~isempty(sol_ref) && isstruct(sol_ref) && isfield(sol_ref, 'V') && isfield(sol_ref, 'theta_all')
    assign_ref = zeros(K, 1);
    if isfield(sol_ref, 'assign') && ~isempty(sol_ref.assign)
        assign_ref = sol_ref.assign(:);
    end
    assign_ref = enforce_capacity_local(assign_ref, L, k0);

    eval_opts = struct('semantic_mode', semantic_mode, 'table_path', table_path);
    cost_matrix = zeros(K, L + 1);

    % Baseline: force user k to be direct (others fixed)
    for k = 1:K
        assign_tmp = assign_ref;
        assign_tmp(k) = 0;
        sol_tmp = struct('assign', assign_tmp, 'theta_all', sol_ref.theta_all, 'V', sol_ref.V);
        out_tmp = evaluate_system_rsma(cfg, ch, geom, sol_tmp, profile, eval_opts);
        bmul = 1 + (urgent_mult - 1) * double(urgent_mask(k));
        vcost = vio_bias_d * double(out_tmp.delay_vio_vec(k)) + vio_bias_s * double(out_tmp.semantic_vio_vec(k));
        cost_matrix(k, 1) = out_tmp.qoe_vec(k) + bmul * vcost;
    end

    % Candidate RIS: force user k to RIS l (others fixed)
    for l = 1:L
        for k = 1:K
            assign_tmp = assign_ref;
            assign_tmp(k) = l;
            sol_tmp = struct('assign', assign_tmp, 'theta_all', sol_ref.theta_all, 'V', sol_ref.V);
            out_tmp = evaluate_system_rsma(cfg, ch, geom, sol_tmp, profile, eval_opts);
            bmul = 1 + (urgent_mult - 1) * double(urgent_mask(k));
            vcost = vio_bias_d * double(out_tmp.delay_vio_vec(k)) + vio_bias_s * double(out_tmp.semantic_vio_vec(k));
            cost_matrix(k, l + 1) = out_tmp.qoe_vec(k) + bmul * vcost;
        end
    end

    gain_delta = cost_matrix(:, 1) - cost_matrix(:, 2:end);
    assign = gain_matching(gain_delta, k0);
    return;
end

% cost_matrix(k, l+1): user k cost with direct(l=0) / RIS l>=1
cost_matrix = zeros(K, L + 1);

% Direct-link QoE cost
for k = 1:K
    hk = ch.h_d(:, k);
    g = norm(hk)^2;
    gamma_base = p_per_user * g / (cfg.noise_watts + cfg.eps);
    xi_base = semantic_map(gamma_base, profile.M_k(k), semantic_mode, table_path, sem_params);
    rate_base = cfg.bandwidth * log2(1 + gamma_base + cfg.eps);
    prop_delay_base = 0;

    [avg_q, ~, ~] = qoe( ...
        cfg, gamma_base, profile.M_k(k), xi_base, profile.weights(k, :), prop_delay_base, ...
        rate_base, profile.d_k(k), profile.dmax_k(k));
    delay_vio = double((cfg.rho .* profile.M_k(k) / (rate_base + cfg.eps) + prop_delay_base) > profile.d_k(k));
    sem_vio = double((1 - xi_base) > profile.dmax_k(k));
    bmul = 1 + (urgent_mult - 1) * double(urgent_mask(k));
    cost_matrix(k, 1) = avg_q + bmul * (vio_bias_d * delay_vio + vio_bias_s * sem_vio);
end

% RIS-link QoE cost
for l = 1:L
    for k = 1:K
        h_ris = ch.H_ris(:, k, l);
        G_l = ch.G(:, :, l);

        if isfield(cfg, 'ris_phase_mode') && strcmpi(cfg.ris_phase_mode, 'align')
            hdk = ch.h_d(:, k);
            denom = norm(hdk) + cfg.eps;
            w0 = hdk / denom;
            proj = (w0' * G_l).' .* h_ris;
            theta_kl = exp(-1j * angle(proj + cfg.eps));
        else
            theta_kl = ch.theta(:, l);
        end

        h_eff = ch.h_d(:, k) + cfg.ris_gain * (G_l * (theta_kl .* h_ris));
        g = norm(h_eff)^2;
        gamma_ris = p_per_user * g / (cfg.noise_watts + cfg.eps);
        xi_ris = semantic_map(gamma_ris, profile.M_k(k), semantic_mode, table_path, sem_params);
        rate_ris = cfg.bandwidth * log2(1 + gamma_ris + cfg.eps);
        prop_delay_ris = local_prop_delay_ris(geom, k, l, cfg);

        [avg_q, ~, ~] = qoe( ...
            cfg, gamma_ris, profile.M_k(k), xi_ris, profile.weights(k, :), prop_delay_ris, ...
            rate_ris, profile.d_k(k), profile.dmax_k(k));
        delay_vio = double((cfg.rho .* profile.M_k(k) / (rate_ris + cfg.eps) + prop_delay_ris) > profile.d_k(k));
        sem_vio = double((1 - xi_ris) > profile.dmax_k(k));
        bmul = 1 + (urgent_mult - 1) * double(urgent_mask(k));
        cost_matrix(k, l + 1) = avg_q + bmul * (vio_bias_d * delay_vio + vio_bias_s * sem_vio);
    end
end

% Gain: positive means RIS is better than direct
gain_delta = cost_matrix(:, 1) - cost_matrix(:, 2:end);

assign = gain_matching(gain_delta, k0);

end

function assign = gain_matching(gain_delta, k0)
%GAIN_MATCHING One-to-many matching with capacity and gain replacement.
K = size(gain_delta, 1);
L = size(gain_delta, 2);

assign = zeros(K, 1);
capacity = k0 * ones(L, 1);
assigned = false(K, 1);

preference_list = cell(K, 1);
for k = 1:K
    gains = gain_delta(k, :);
    valid_ris = find(gains > 0);
    if ~isempty(valid_ris)
        [~, idx] = sort(gains(valid_ris), 'descend');
        preference_list{k} = valid_ris(idx);
    end
end

max_iter = K * L;
iter = 0;
while iter < max_iter
    iter = iter + 1;

    free_user = -1;
    for k = 1:K
        if ~assigned(k) && ~isempty(preference_list{k})
            free_user = k;
            break;
        end
    end
    if free_user < 0
        break;
    end

    ris = preference_list{free_user}(1);
    preference_list{free_user}(1) = [];

    if capacity(ris) > 0
        assign(free_user) = ris;
        assigned(free_user) = true;
        capacity(ris) = capacity(ris) - 1;
    else
        current_users = find(assign == ris);
        if isempty(current_users)
            assign(free_user) = ris;
            assigned(free_user) = true;
        else
            min_gain = inf;
            worst_user = -1;
            for u = current_users'
                g = gain_delta(u, ris);
                if g < min_gain
                    min_gain = g;
                    worst_user = u;
                end
            end

            if gain_delta(free_user, ris) > min_gain
                assign(worst_user) = 0;
                assigned(worst_user) = false;
                assign(free_user) = ris;
                assigned(free_user) = true;
            else
                assign(free_user) = 0;
                assigned(free_user) = true;
            end
        end
    end
end

for k = 1:K
    if ~assigned(k)
        assign(k) = 0;
    end
end
end

function assign = enforce_capacity_local(assign, L, k0)
assign = assign(:);
for l = 1:L
    idx = find(assign == l);
    if numel(idx) > k0
        drop_idx = idx((k0 + 1):end);
        assign(drop_idx) = 0;
    end
end
end

function profile = build_local_profile(cfg, weights, profile_in, geom)
K = cfg.num_users;

if nargin >= 3 && ~isempty(profile_in) && isstruct(profile_in)
    profile = profile_in;
else
    profile = build_profile_urgent_normal(cfg, geom, struct());
end

if ~isfield(profile, 'M_k') || isempty(profile.M_k)
    profile.M_k = cfg.m_k * ones(K, 1);
end
if numel(profile.M_k) == 1
    profile.M_k = repmat(profile.M_k, K, 1);
else
    profile.M_k = profile.M_k(:);
end

if ~isfield(profile, 'd_k') || isempty(profile.d_k)
    if isfield(profile, 'groups') && isfield(profile.groups, 'urgent_idx') && isfield(profile.groups, 'normal_idx')
        uidx = normalize_index_vector(profile.groups.urgent_idx, K);
        nidx = normalize_index_vector(profile.groups.normal_idx, K);
        profile.d_k = cfg.deadlines(min(2, end)) * ones(K, 1);
        profile.d_k(uidx) = cfg.deadlines(1);
        profile.d_k(nidx) = cfg.deadlines(min(2, end));
    else
        profile.d_k = cfg.deadlines(1) * ones(K, 1);
    end
elseif numel(profile.d_k) == 1
    profile.d_k = repmat(profile.d_k, K, 1);
else
    profile.d_k = profile.d_k(:);
end

if ~isfield(profile, 'dmax_k') || isempty(profile.dmax_k)
    profile.dmax_k = cfg.dmax * ones(K, 1);
elseif numel(profile.dmax_k) == 1
    profile.dmax_k = repmat(profile.dmax_k, K, 1);
else
    profile.dmax_k = profile.dmax_k(:);
end

if ~isfield(profile, 'weights') || isempty(profile.weights)
    w = weights;
else
    w = profile.weights;
end

if size(w, 1) == 1 && size(w, 2) == 2
    profile.weights = repmat(w, K, 1);
elseif size(w, 1) == K && size(w, 2) == 2
    profile.weights = w;
else
    error('weights must be 1x2 or Kx2.');
end
end

function pd = local_prop_delay_ris(geom, k, l, cfg)
pd = 0;
if isempty(geom) || ~isfield(geom, 'bs') || ~isfield(geom, 'ue') || ~isfield(geom, 'ris')
    return;
end
if l <= 0 || l > size(geom.ris, 1) || k > size(geom.ue, 1)
    return;
end

factor = 1e-5;
if isfield(cfg, 'prop_delay_factor') && ~isempty(cfg.prop_delay_factor)
    factor = cfg.prop_delay_factor;
end

bs = geom.bs;
if size(bs, 1) > 1, bs = bs(1, :); end
bs = bs(1:2);
ris = geom.ris(l, 1:2);
ue = geom.ue(k, 1:2);

pd = factor * (norm(ris - bs) + norm(ue - ris));
end

function sem_params = local_semantic_params(cfg)
if isfield(cfg, 'proxy_a') && isfield(cfg, 'proxy_b')
    sem_params = struct('a', cfg.proxy_a, 'b', cfg.proxy_b);
else
    sem_params = struct();
end
end

function v = get_field_default(s, f, d)
if isfield(s, f) && ~isempty(s.(f))
    v = s.(f);
else
    v = d;
end
end

