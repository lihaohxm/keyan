function [avg_qoe, qoe_vec, meta] = qoe(cfg, gamma_sem, M, xi, weights, prop_delay, rate_bps, d_vec, dmax_vec)
%QOE Compute per-user QoE cost (lower is better).
% Compatible with:
%   old: qoe(cfg, gamma_sem, M, xi, weights, prop_delay)
%   new: qoe(cfg, gamma_sem, M, xi, weights, prop_delay, rate_bps, d_vec, dmax_vec)

K = numel(gamma_sem);
gamma_sem = gamma_sem(:);
xi = xi(:);

if numel(M) == 1
    M = repmat(M, K, 1);
else
    M = M(:);
end

if nargin < 5 || isempty(weights)
    weights = cfg.weights(1, :);
end
if size(weights, 1) == 1 && size(weights, 2) == 2
    weights = repmat(weights, K, 1);
elseif ~(size(weights, 1) == K && size(weights, 2) == 2)
    error('weights must be 1x2 or Kx2.');
end
w_d = weights(:, 1);
w_s = weights(:, 2);

if nargin < 6 || isempty(prop_delay)
    prop_delay = zeros(K, 1);
else
    prop_delay = prop_delay(:);
end

% Transmission delay term (old/new compatible)
if nargin < 7 || isempty(rate_bps)
    T_tx_only = cfg.rho .* M ./ (cfg.bandwidth .* log2(1 + gamma_sem) + cfg.eps);
else
    rate_bps = rate_bps(:);
    T_tx_only = cfg.rho .* M ./ (rate_bps + cfg.eps);
end
T_tx = T_tx_only + prop_delay;

% Semantic distortion
D = 1 - xi;

% Deadlines
if nargin >= 8 && ~isempty(d_vec)
    d_vec = d_vec(:);
else
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

% Semantic distortion bounds
if nargin >= 9 && ~isempty(dmax_vec)
    if numel(dmax_vec) == 1
        dmax_vec = repmat(dmax_vec, K, 1);
    else
        dmax_vec = dmax_vec(:);
    end
else
    dmax_vec = cfg.dmax * ones(K, 1);
end

% Ratios (bounded QoE definition):
% - QoE-related base terms Qd/Qs are sigmoid in [0,1]
% - exceedance penalties are tracked separately in meta for diagnostics
ratio_d = T_tx ./ d_vec;
ratio_s = D ./ dmax_vec;

if isfield(cfg, 'max_ratio_d') && ~isempty(cfg.max_ratio_d) && cfg.max_ratio_d > 0
    ratio_d = min(ratio_d, cfg.max_ratio_d);
end
if isfield(cfg, 'max_ratio_s') && ~isempty(cfg.max_ratio_s) && cfg.max_ratio_s > 0
    ratio_s = min(ratio_s, cfg.max_ratio_s);
end

% Shift the sigmoid center earlier than the hard limit so moderately tight
% low-power / low-L cases do not collapse toward zero cost too early.
delay_ratio_center = get_cfg_or_default(cfg, 'delay_ratio_center', 1.0);
semantic_ratio_center = get_cfg_or_default(cfg, 'semantic_ratio_center', 1.0);
Qd_sig = 1 ./ (1 + exp(-cfg.b_d .* (ratio_d - delay_ratio_center)));
Qs_sig = 1 ./ (1 + exp(-cfg.b_s .* (ratio_s - semantic_ratio_center)));

penalty_d = cfg.h_d .* max(0, ratio_d - 1);
delay_soft_trigger = get_cfg_or_default(cfg, 'delay_soft_ratio_trigger', 1.0);
delay_soft_penalty = get_cfg_or_default(cfg, 'delay_soft_penalty', 0.0);
if delay_soft_trigger < 1
    delay_soft_scale = max(1 - delay_soft_trigger, cfg.eps);
    penalty_d = penalty_d + delay_soft_penalty .* ...
        (max(0, ratio_d - delay_soft_trigger) ./ delay_soft_scale) .^ 2;
end
semantic_penalty_power = 2.0;
if isfield(cfg, 'semantic_penalty_power') && ~isempty(cfg.semantic_penalty_power)
    semantic_penalty_power = cfg.semantic_penalty_power;
end
penalty_s = cfg.h_s .* max(0, ratio_s - 1) .^ semantic_penalty_power;
semantic_soft_trigger = get_cfg_or_default(cfg, 'semantic_soft_ratio_trigger', 1.0);
semantic_soft_penalty = get_cfg_or_default(cfg, 'semantic_soft_penalty', 0.0);
if semantic_soft_trigger < 1
    semantic_soft_scale = max(1 - semantic_soft_trigger, cfg.eps);
    penalty_s = penalty_s + semantic_soft_penalty .* ...
        (max(0, ratio_s - semantic_soft_trigger) ./ semantic_soft_scale) .^ 2;
end

Qd = Qd_sig + penalty_d; % [Antigravity Fix]
Qs = Qs_sig + penalty_s; % [Antigravity Fix]

if any(~isfinite(Qd)) || any(~isfinite(Qs))
    error('qoe:nonfinite_q', 'Non-finite Qd/Qs encountered.');
end
if any(Qd < -1e-12) || any(Qs < -1e-12)
    error('qoe:q_out_of_range', 'Qd/Qs must be >= 0.');
end

w_sum = w_d + w_s;
if any(w_sum <= cfg.eps) || any(~isfinite(w_sum))
    error('qoe:bad_weights', 'Each user weight sum must be positive and finite.');
end
w_d_n = w_d ./ w_sum;
w_s_n = w_s ./ w_sum;

qoe_vec = w_d_n .* Qd + w_s_n .* Qs;
if any(qoe_vec < -1e-12) || any(~isfinite(qoe_vec))
    error('qoe:qoe_out_of_range', 'QoE must be >= 0 and finite.');
end
avg_qoe = mean(qoe_vec);

meta.T_tx = T_tx;
meta.D = D;
meta.ratio_d = ratio_d;
meta.ratio_s = ratio_s;
meta.delay_violation_rate = mean(T_tx > d_vec);
meta.semantic_violation_rate = mean(D > dmax_vec);
meta.deadline_ok = mean(T_tx <= d_vec);
meta.semantic_ok = mean(D <= dmax_vec);
meta.Qd = Qd;
meta.Qs = Qs;
meta.penalty_d = penalty_d;
meta.penalty_s = penalty_s;
meta.Qd_with_penalty = Qd_sig + penalty_d;
meta.Qs_with_penalty = Qs_sig + penalty_s;

end

function v = get_cfg_or_default(cfg, field_name, default_v)
if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
    v = cfg.(field_name);
else
    v = default_v;
end
end
