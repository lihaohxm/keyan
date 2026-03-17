function cfg_out = attach_profile_context_to_cfg(cfg, profile)
%ATTACH_PROFILE_CONTEXT_TO_CFG Export task structure statistics into cfg.
% This keeps the rsma_wmmse interface unchanged while allowing its
% allocation logic to react to the current task mix.

cfg_out = cfg;

if isempty(profile) || ~isfield(profile, 'weights') || ~isfield(profile, 'd_k') || ...
        ~isfield(profile, 'dmax_k') || ~isfield(profile, 'groups') || ...
        ~isfield(profile.groups, 'urgent_idx')
    return;
end

K = cfg.num_users;
urgent_idx = normalize_index_vector(profile.groups.urgent_idx, K);
if isempty(urgent_idx)
    return;
end

weight_sum = max(sum(profile.weights, 2), cfg.eps);
delay_pref = profile.weights(:, 1) ./ weight_sum;
semantic_pref = profile.weights(:, 2) ./ weight_sum;

delay_pressure = 1 ./ max(profile.d_k(:), cfg.eps);
semantic_pressure = 1 ./ max(profile.dmax_k(:), cfg.eps);
delay_pressure = delay_pressure / max(max(delay_pressure), cfg.eps);
semantic_pressure = semantic_pressure / max(max(semantic_pressure), cfg.eps);

cfg_out.task_delay_pref_mean = mean(delay_pref);
cfg_out.task_semantic_pref_mean = mean(semantic_pref);
cfg_out.task_delay_pressure_mean = mean(delay_pressure);
cfg_out.task_semantic_pressure_mean = mean(semantic_pressure);

cfg_out.urgent_delay_pref_mean = mean(delay_pref(urgent_idx));
cfg_out.urgent_semantic_pref_mean = mean(semantic_pref(urgent_idx));
cfg_out.urgent_delay_pressure_mean = mean(delay_pressure(urgent_idx));
cfg_out.urgent_semantic_pressure_mean = mean(semantic_pressure(urgent_idx));
cfg_out.urgent_fraction = numel(urgent_idx) / max(K, 1);
end
