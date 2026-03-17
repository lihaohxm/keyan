function [cost_v, info] = semantic_compare_cost(cfg, out_s)
%SEMANTIC_COMPARE_COST Semantic-aware comparison cost for AO acceptance.
% Combines the base QoE cost with urgent semantic violation/distortion
% penalties so that candidate updates do not improve QoE by sacrificing
% semantic reliability.

if nargin < 1 || isempty(cfg)
    cfg = struct();
end

base_cost = get_base_cost(out_s);
urgent_sem_vio = get_metric(out_s, 'urgent_semantic_violation', ...
    get_metric(out_s, 'semantic_vio_rate_all', NaN));
urgent_sem_dist = get_metric(out_s, 'urgent_semantic_distortion', NaN);
if ~isfinite(urgent_sem_dist)
    urgent_sem_dist = get_metric(out_s, 'semantic_distortion_mean_all', NaN);
end
if ~isfinite(urgent_sem_dist)
    xi_mean = get_metric(out_s, 'xi_mean_all', NaN);
    if isfinite(xi_mean)
        urgent_sem_dist = 1 - xi_mean;
    end
end

if ~get_cfg(cfg, 'semantic_guard_enable', true)
    vio_pen = 0;
    dist_pen = 0;
else
    vio_target = get_cfg(cfg, 'semantic_guard_vio_target', 0.25);
    dist_target = get_cfg(cfg, 'semantic_guard_dist_target', get_cfg(cfg, 'dmax', 0.45));
    vio_weight = get_cfg(cfg, 'semantic_guard_vio_weight', 0.35);
    dist_weight = get_cfg(cfg, 'semantic_guard_dist_weight', 0.20);
    vio_pen = vio_weight * max(0, urgent_sem_vio - vio_target);
    dist_pen = dist_weight * max(0, urgent_sem_dist - dist_target);
end

cost_v = base_cost + vio_pen + dist_pen;

info = struct();
info.base_cost = base_cost;
info.urgent_semantic_violation = urgent_sem_vio;
info.urgent_semantic_distortion = urgent_sem_dist;
info.violation_penalty = vio_pen;
info.distortion_penalty = dist_pen;
info.penalty = vio_pen + dist_pen;
end

function cost_v = get_base_cost(out_s)
if isfield(out_s, 'composite_cost') && ~isempty(out_s.composite_cost) && isfinite(out_s.composite_cost)
    cost_v = out_s.composite_cost;
elseif isfield(out_s, 'avg_qoe_pure') && ~isempty(out_s.avg_qoe_pure) && isfinite(out_s.avg_qoe_pure)
    cost_v = out_s.avg_qoe_pure;
else
    cost_v = out_s.avg_qoe;
end
end

function value = get_metric(s, name, default_value)
if isfield(s, name) && ~isempty(s.(name))
    value = s.(name);
else
    value = default_value;
end
if ~isscalar(value)
    value = mean(value(:), 'omitnan');
end
end

function value = get_cfg(cfg, name, default_value)
if isfield(cfg, name) && ~isempty(cfg.(name))
    value = cfg.(name);
else
    value = default_value;
end
end
