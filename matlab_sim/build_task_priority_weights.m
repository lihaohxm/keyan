function [q_weights, meta] = build_task_priority_weights(cfg, out_prev, profile, opts)
%BUILD_TASK_PRIORITY_WEIGHTS Build endogenous per-user weights for WMMSE.
% Higher weights are assigned to users whose tasks are stricter and whose
% current delay/semantic states are closer to or beyond their limits.

if nargin < 4 || isempty(opts)
    opts = struct();
end

K = cfg.num_users;
baseline_mode = get_opt(opts, 'baseline_mode', false);
use_gradient_term = get_opt(opts, 'use_gradient_term', true);

if baseline_mode
    q_weights = ones(K, 1);
    meta = struct('mode', 'baseline');
    return;
end

urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
is_urgent = false(K, 1);
is_urgent(urgent_idx) = true;

priority_score = safe_vec(profile, 'priority_score', K, 1);
priority_score = normalize_vec(priority_score);

weight_sum = max(sum(profile.weights, 2), cfg.eps);
delay_pref = profile.weights(:, 1) ./ weight_sum;
semantic_pref = profile.weights(:, 2) ./ weight_sum;

delay_pressure = normalize_vec(1 ./ max(profile.d_k(:), cfg.eps));
semantic_pressure = normalize_vec(1 ./ max(profile.dmax_k(:), cfg.eps));
task_pressure = normalize_vec(delay_pref .* delay_pressure + semantic_pref .* semantic_pressure);

T_tx = safe_vec(out_prev, 'T_tx', K, 0);
D_vec = safe_vec(out_prev, 'D', K, 0);
delay_tightness = T_tx ./ max(profile.d_k(:), cfg.eps);
semantic_tightness = D_vec ./ max(profile.dmax_k(:), cfg.eps);
delay_excess = max(0, T_tx - profile.d_k(:)) ./ max(profile.d_k(:), cfg.eps);
semantic_excess = max(0, D_vec - profile.dmax_k(:)) ./ max(profile.dmax_k(:), cfg.eps);
delay_excess = min(delay_excess, get_cfg(cfg, 'qweight_excess_cap', 2.0));
semantic_excess = min(semantic_excess, get_cfg(cfg, 'qweight_excess_cap', 2.0));
delay_tightness = min(delay_tightness, get_cfg(cfg, 'qweight_tightness_cap', 1.5));
semantic_tightness = min(semantic_tightness, get_cfg(cfg, 'qweight_tightness_cap', 1.5));
delay_tightness_score = normalize_vec(delay_pref .* delay_tightness);
semantic_tightness_score = normalize_vec(semantic_pref .* semantic_tightness);

gradient_score = zeros(K, 1);
if use_gradient_term
    Qd = safe_vec(out_prev, 'Qd_vec', K, 0.5);
    Qs = safe_vec(out_prev, 'Qs_vec', K, 0.5);
    rate_bps = safe_vec(out_prev, 'rate_vec_bps', K, cfg.bandwidth);
    sigmoid_floor = get_cfg(cfg, 'wmmse_floor', 0.05);

    grad_d = max(Qd, sigmoid_floor) .* max(1 - Qd, sigmoid_floor);
    grad_s = max(Qs, sigmoid_floor) .* max(1 - Qs, sigmoid_floor);

    dQdT = delay_pref .* (cfg.b_d ./ max(profile.d_k(:), cfg.eps)) .* grad_d;
    dQdD = semantic_pref .* (cfg.b_s ./ max(profile.dmax_k(:), cfg.eps)) .* grad_s;
    dTdR = T_tx ./ max(rate_bps, cfg.eps);
    dDdR = D_vec ./ max(rate_bps, cfg.eps);

    gradient_score = normalize_vec(dQdT .* dTdR + dQdD .* dDdR);
end

urgent_state_scale = get_urgent_state_scale(cfg, out_prev, urgent_idx, delay_tightness, semantic_tightness);
urgent_bonus = zeros(K, 1);
urgent_bonus(is_urgent) = get_cfg(cfg, 'qweight_urgent_bonus', 0.35) .* urgent_state_scale;

q_weights = 1 ...
    + urgent_bonus ...
    + urgent_state_scale .* get_cfg(cfg, 'qweight_priority_score_gain', 0.25) .* priority_score ...
    + urgent_state_scale .* get_cfg(cfg, 'qweight_task_pressure_gain', 1.00) .* task_pressure ...
    + urgent_state_scale .* get_cfg(cfg, 'qweight_delay_tightness_gain', 0.15) .* delay_tightness_score ...
    + urgent_state_scale .* get_cfg(cfg, 'qweight_semantic_tightness_gain', 0.80) .* semantic_tightness_score ...
    + get_cfg(cfg, 'qweight_delay_excess_gain', 0.35) .* (delay_pref .* delay_excess) ...
    + get_cfg(cfg, 'qweight_semantic_excess_gain', 0.90) .* (semantic_pref .* semantic_excess) ...
    + urgent_state_scale .* get_cfg(cfg, 'qweight_gradient_gain', 0.30) .* gradient_score;

q_weights = min(q_weights, get_cfg(cfg, 'qweight_max', 4.0));
q_weights = max(q_weights, get_cfg(cfg, 'qweight_min', 0.6));

m = mean(q_weights);
if ~isfinite(m) || m <= cfg.eps || any(~isfinite(q_weights))
    q_weights = ones(K, 1);
else
    q_weights = q_weights / (m + cfg.eps);
end

meta = struct();
meta.delay_pref = delay_pref;
meta.semantic_pref = semantic_pref;
meta.task_pressure = task_pressure;
meta.priority_score = priority_score;
meta.delay_tightness = delay_tightness;
meta.semantic_tightness = semantic_tightness;
meta.delay_tightness_score = delay_tightness_score;
meta.semantic_tightness_score = semantic_tightness_score;
meta.delay_excess = delay_excess;
meta.semantic_excess = semantic_excess;
meta.gradient_score = gradient_score;
meta.urgent_idx = urgent_idx;
meta.urgent_state_scale = urgent_state_scale;
end

function scale = get_urgent_state_scale(cfg, out_prev, urgent_idx, delay_tightness, semantic_tightness)
bias_floor = get_cfg(cfg, 'qweight_bias_floor', 0.35);
stress_trigger = get_cfg(cfg, 'qweight_stress_trigger', 0.85);

if isempty(urgent_idx)
    scale = bias_floor;
    return;
end

urgent_delay_vio = safe_scalar(out_prev, 'urgent_delay_violation', 1.0);
urgent_sem_vio = safe_scalar(out_prev, 'urgent_semantic_violation', 1.0);

delay_tight_urgent = mean(delay_tightness(urgent_idx));
semantic_tight_urgent = mean(semantic_tightness(urgent_idx));
delay_tight_score = smooth_excess(delay_tight_urgent, stress_trigger);
semantic_tight_score = smooth_excess(semantic_tight_urgent, stress_trigger);

stress = max([urgent_delay_vio, urgent_sem_vio, delay_tight_score, semantic_tight_score]);
stress = min(max(stress, 0), 1);
scale = bias_floor + (1 - bias_floor) * stress;
end

function score = smooth_excess(value, trigger)
if ~isfinite(value)
    score = 1.0;
    return;
end
if value <= trigger
    score = 0.0;
    return;
end
score = min(max((value - trigger) / max(1 - trigger, 1e-12), 0), 1);
end

function value = safe_scalar(s, name, defaultv)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name)) && isfinite(s.(name))
    value = double(s.(name));
else
    value = defaultv;
end
end

function v = safe_vec(s, name, K, defaultv)
if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
    raw = s.(name);
    if numel(raw) == 1
        v = repmat(raw, K, 1);
    elseif numel(raw) == K
        v = raw(:);
    else
        v = defaultv * ones(K, 1);
    end
else
    v = defaultv * ones(K, 1);
end
end

function out = normalize_vec(x)
x = double(x(:));
scale = max(abs(x));
if ~isfinite(scale) || scale <= 0
    out = ones(size(x));
else
    out = x / scale;
end
end

function v = get_cfg(cfg, name, default_v)
if isfield(cfg, name) && ~isempty(cfg.(name))
    v = cfg.(name);
else
    v = default_v;
end
end

function v = get_opt(opts, name, default_v)
if isfield(opts, name) && ~isempty(opts.(name))
    v = opts.(name);
else
    v = default_v;
end
end
