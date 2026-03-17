function profile = build_profile_urgent_normal(cfg, geom, opts)
%BUILD_PROFILE_URGENT_NORMAL Build task profile and select high-priority users.
% Default behavior:
% 1) Geometry remains task-agnostic.
% 2) Each user samples a task type independently of location.
% 3) High-priority users are selected by task pressure and pre-scheduling
%    channel risk, then stored in profile.groups.{urgent_idx,normal_idx}.

if nargin < 3 || isempty(opts)
    opts = struct();
end

K = cfg.num_users;

if has_explicit_groups(geom, opts)
    profile = build_profile_from_explicit_groups(cfg, geom, opts, K);
    return;
end

num_task_types = numel(cfg.task_type_probs);
task_idx = resolve_task_types(cfg, geom, opts, K, num_task_types);
task_names = get_task_names(cfg, num_task_types);

profile.task_type_idx = task_idx(:);
profile.task_type_name = task_names(task_idx(:));
profile.M_k = task_param_vector(cfg, 'task_M', task_idx, cfg.m_k);
profile.d_k = task_param_vector(cfg, 'task_deadlines', task_idx, cfg.deadlines(1));
profile.dmax_k = task_param_vector(cfg, 'task_dmax', task_idx, cfg.dmax);

weight_rows = round(task_param_vector(cfg, 'task_weight_rows', task_idx, 1));
weight_rows = min(max(weight_rows, 1), size(cfg.weights, 1));
profile.weight_row_idx = weight_rows(:);
profile.weights = cfg.weights(weight_rows, :);

if isfield(opts, 'M_all') && ~isempty(opts.M_all)
    profile.M_k = as_kvec(opts.M_all, K, 'opts.M_all');
end
if isfield(opts, 'dmax_all') && ~isempty(opts.dmax_all)
    profile.dmax_k = as_kvec(opts.dmax_all, K, 'opts.dmax_all');
end
if isfield(opts, 'd_k_all') && ~isempty(opts.d_k_all)
    profile.d_k = as_kvec(opts.d_k_all, K, 'opts.d_k_all');
end
if isfield(opts, 'weights') && ~isempty(opts.weights)
    profile.weights = as_weight_matrix(opts.weights, K);
end

[channel_quality_proxy, channel_risk] = compute_channel_priority_proxy(cfg, geom, K);
weight_sum = max(sum(profile.weights, 2), cfg.eps);
delay_pref = profile.weights(:, 1) ./ weight_sum;
semantic_pref = profile.weights(:, 2) ./ weight_sum;

pref_gain = get_opt(opts, 'priority_preference_gain', get_opt(cfg, 'priority_preference_gain', 0.75));
delay_pressure = normalize_component((1 ./ max(profile.d_k, cfg.eps)) .* (1 + pref_gain .* delay_pref));
semantic_pressure = normalize_component((1 ./ max(profile.dmax_k, cfg.eps)) .* (1 + pref_gain .* semantic_pref));

channel_risk_norm = normalize_component(channel_risk);
channel_softness = get_opt(opts, 'priority_channel_softness', get_opt(cfg, 'priority_channel_softness', 0.50));
channel_risk_soft = channel_risk_norm ./ (channel_risk_norm + channel_softness + cfg.eps);
channel_risk_score = normalize_component(channel_risk_soft);

priority_weights = get_opt(opts, 'priority_weights', get_opt(cfg, 'priority_weights', [0.50 0.40 0.10]));
priority_weights = normalize_priority_weights(priority_weights);
priority_score = priority_weights(1) * delay_pressure + ...
                 priority_weights(2) * semantic_pressure + ...
                 priority_weights(3) * channel_risk_score;

urgent_count = min(K, max(0, round(get_opt(opts, 'num_urgent', get_opt(cfg, 'num_urgent', min(K, 4))))));
[~, order] = sort(priority_score, 'descend');
urgent_idx = sort(order(1:urgent_count));
normal_idx = setdiff((1:K).', urgent_idx(:), 'stable');

profile.channel_quality_proxy = channel_quality_proxy(:);
profile.channel_risk = channel_risk(:);
profile.delay_pref = delay_pref(:);
profile.semantic_pref = semantic_pref(:);
profile.priority_components = struct( ...
    'delay_pressure', delay_pressure(:), ...
    'semantic_pressure', semantic_pressure(:), ...
    'channel_risk', channel_risk_score(:));
profile.priority_score = priority_score(:);
profile.priority_weights = priority_weights(:).';
profile.groups.urgent_idx = urgent_idx(:);
profile.groups.normal_idx = normal_idx(:);
end

function tf = has_explicit_groups(geom, opts)
tf = false;
if isfield(opts, 'urgent_idx') || isfield(opts, 'normal_idx')
    tf = true;
    return;
end
ut = extract_user_type(geom, opts);
if isempty(ut)
    return;
end
ut = lower(strtrim(ut));
tf = any(ut == "urgent") || any(ut == "normal");
end

function profile = build_profile_from_explicit_groups(cfg, geom, opts, K)
urgent_weight_row = get_opt(opts, 'urgent_weight_row', 1);
normal_weight_row = get_opt(opts, 'normal_weight_row', min(2, size(cfg.weights, 1)));
urgent_deadline = get_opt(opts, 'urgent_deadline', cfg.deadlines(1));
normal_deadline = get_opt(opts, 'normal_deadline', cfg.deadlines(min(2, end)));
dmax_all = get_opt(opts, 'dmax_all', cfg.dmax);
M_all = get_opt(opts, 'M_all', cfg.m_k);

[urgent_idx, normal_idx] = resolve_explicit_groups(K, geom, opts);

profile.M_k = as_kvec(M_all, K, 'M_all');

num_rows = size(cfg.weights, 1);
urgent_weight_row = min(max(round(urgent_weight_row), 1), num_rows);
normal_weight_row = min(max(round(normal_weight_row), 1), num_rows);

profile.weights = zeros(K, 2);
profile.weights(urgent_idx, :) = repmat(cfg.weights(urgent_weight_row, :), numel(urgent_idx), 1);
profile.weights(normal_idx, :) = repmat(cfg.weights(normal_weight_row, :), numel(normal_idx), 1);
profile.weight_row_idx = zeros(K, 1);
profile.weight_row_idx(urgent_idx) = urgent_weight_row;
profile.weight_row_idx(normal_idx) = normal_weight_row;

profile.d_k = zeros(K, 1);
profile.d_k(urgent_idx) = urgent_deadline;
profile.d_k(normal_idx) = normal_deadline;
profile.dmax_k = as_kvec(dmax_all, K, 'dmax_all');
profile.task_type_idx = ones(K, 1);
profile.task_type_name = repmat("legacy_grouped", K, 1);
profile.priority_score = zeros(K, 1);
profile.channel_quality_proxy = nan(K, 1);
profile.channel_risk = nan(K, 1);
profile.priority_components = struct( ...
    'delay_pressure', nan(K, 1), ...
    'semantic_pressure', nan(K, 1), ...
    'channel_risk', nan(K, 1));
profile.priority_weights = [1 0 0];
profile.groups.urgent_idx = urgent_idx(:);
profile.groups.normal_idx = normal_idx(:);
end

function [urgent_idx, normal_idx] = resolve_explicit_groups(K, geom, opts)
if isfield(opts, 'urgent_idx') || isfield(opts, 'normal_idx')
    urgent_idx = normalize_index_vector(get_opt(opts, 'urgent_idx', []), K);
    normal_idx = normalize_index_vector(get_opt(opts, 'normal_idx', []), K);
else
    ut = extract_user_type(geom, opts);
    if isempty(ut)
        error('build_profile_urgent_normal:missing_group_source', ...
            'Explicit groups require opts.{urgent_idx,normal_idx} or user_type labels.');
    end
    n = min(numel(ut), K);
    ut = lower(strtrim(ut(1:n)));
    if n < K
        ut = [ut; repmat("normal", K - n, 1)];
    end
    urgent_idx = normalize_index_vector(ut == "urgent", K);
    normal_idx = normalize_index_vector(ut == "normal", K);
end

normal_idx = setdiff(normal_idx(:), urgent_idx(:), 'stable');
covered = false(K, 1);
covered(urgent_idx) = true;
covered(normal_idx) = true;
normal_idx = [normal_idx; find(~covered)];

urgent_idx = unique(urgent_idx(:), 'stable');
normal_idx = unique(normal_idx(:), 'stable');
end

function ut = extract_user_type(geom, opts)
ut = strings(0, 1);
if isfield(opts, 'user_type') && ~isempty(opts.user_type)
    ut = to_strvec(opts.user_type);
elseif nargin >= 1 && ~isempty(geom) && isstruct(geom) && isfield(geom, 'user_type') && ~isempty(geom.user_type)
    ut = to_strvec(geom.user_type);
end
end

function task_idx = resolve_task_types(cfg, geom, opts, K, num_task_types)
if isfield(opts, 'task_type_idx') && ~isempty(opts.task_type_idx)
    task_idx = as_kvec(opts.task_type_idx, K, 'opts.task_type_idx');
    task_idx = round(task_idx);
    task_idx = min(max(task_idx, 1), num_task_types);
    return;
end

task_probs = cfg.task_type_probs(:);
if isempty(task_probs) || any(task_probs < 0)
    error('cfg.task_type_probs must be a nonnegative vector.');
end
if sum(task_probs) <= 0
    error('cfg.task_type_probs must sum to a positive value.');
end
task_probs = task_probs / sum(task_probs);
task_edges = cumsum(task_probs);

saved_rng = rng;
cleanup_obj = onCleanup(@() rng(saved_rng));
task_seed = get_task_seed(cfg, geom, opts);
rng(task_seed, 'twister');

draws = rand(K, 1);
task_idx = zeros(K, 1);
for k = 1:K
    task_idx(k) = find(draws(k) <= task_edges, 1, 'first');
end

clear cleanup_obj
rng(saved_rng);
end

function seed = get_task_seed(cfg, geom, opts)
if isfield(opts, 'task_seed') && ~isempty(opts.task_seed)
    seed = double(opts.task_seed);
    return;
end
base_seed = 1;
if nargin >= 2 && ~isempty(geom) && isstruct(geom) && isfield(geom, 'seed') && ~isempty(geom.seed)
    base_seed = double(geom.seed);
end
seed = base_seed + get_opt(cfg, 'task_seed_offset', 7919);
end

function v = task_param_vector(cfg, field_name, task_idx, default_v)
if isfield(cfg, field_name) && ~isempty(cfg.(field_name))
    table = cfg.(field_name);
else
    table = default_v;
end

if numel(table) == 1
    v = repmat(table, numel(task_idx), 1);
    return;
end

table = table(:);
if max(task_idx) > numel(table)
    error('cfg.%s must provide at least one entry per task type.', field_name);
end
v = table(task_idx(:));
end

function [quality_proxy, risk_proxy] = compute_channel_priority_proxy(cfg, geom, K)
if isempty(geom) || ~isstruct(geom) || ~isfield(geom, 'bs') || ~isfield(geom, 'ue') || isempty(geom.ue)
    quality_proxy = ones(K, 1);
    risk_proxy = ones(K, 1);
    return;
end

bs = geom.bs(1, 1:2);
ue = geom.ue(:, 1:2);
if size(ue, 1) < K
    error('geometry contains fewer users than cfg.num_users.');
end
ue = ue(1:K, :);

alpha_direct = get_opt(cfg, 'pathloss_exp_direct', get_opt(cfg, 'pathloss_exp', 3.2));
alpha_br = get_opt(cfg, 'pathloss_exp_br', 2.2);
alpha_ru = get_opt(cfg, 'pathloss_exp_ru', 2.2);
ris_weight = get_opt(cfg, 'priority_ris_weight', get_opt(cfg, 'ris_gain', 1));

d_bs = sqrt(sum((ue - bs).^2, 2)) + cfg.eps;
direct_gain = d_bs .^ (-alpha_direct);

best_ris_gain = zeros(K, 1);
if isfield(geom, 'ris') && ~isempty(geom.ris)
    ris = geom.ris(:, 1:2);
    d_br = sqrt(sum((ris - bs).^2, 2)) + cfg.eps;
    br_gain = d_br .^ (-alpha_br);
    for k = 1:K
        d_ru = sqrt(sum((ris - ue(k, :)).^2, 2)) + cfg.eps;
        ru_gain = d_ru .^ (-alpha_ru);
        cascaded_gain = br_gain .* ru_gain;
        best_ris_gain(k) = max(cascaded_gain);
    end
end

quality_proxy = direct_gain + ris_weight * best_ris_gain;
risk_proxy = 1 ./ max(quality_proxy, cfg.eps);
end

function weights = normalize_priority_weights(weights)
weights = double(weights(:).');
if numel(weights) ~= 3
    error('priority_weights must contain exactly three entries.');
end
weights = max(weights, 0);
if sum(weights) <= 0
    weights = [1 0 0];
else
    weights = weights / sum(weights);
end
end

function out = normalize_component(x)
x = double(x(:));
scale = max(abs(x));
if ~isfinite(scale) || scale <= 0
    out = ones(size(x));
else
    out = x / scale;
end
end

function out = get_task_names(cfg, num_types)
if isfield(cfg, 'task_names') && ~isempty(cfg.task_names)
    out = string(cfg.task_names(:));
else
    out = "task_" + string((1:num_types).');
end
if numel(out) < num_types
    for idx = (numel(out) + 1):num_types
        out(idx, 1) = "task_" + string(idx);
    end
end
end

function out = to_strvec(x)
if isstring(x)
    out = x(:);
elseif iscell(x)
    out = string(x(:));
else
    out = strings(0, 1);
end
end

function v = as_kvec(x, K, name)
if numel(x) == 1
    v = repmat(x, K, 1);
elseif numel(x) == K
    v = x(:);
else
    error('%s must be scalar or Kx1.', name);
end
end

function weights = as_weight_matrix(x, K)
if size(x, 1) == 1 && size(x, 2) == 2
    weights = repmat(x, K, 1);
elseif size(x, 1) == K && size(x, 2) == 2
    weights = x;
else
    error('weights override must be 1x2 or Kx2.');
end
end

function v = get_opt(opts, field_name, default_v)
if isfield(opts, field_name) && ~isempty(opts.(field_name))
    v = opts.(field_name);
else
    v = default_v;
end
end
