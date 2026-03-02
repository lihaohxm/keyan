function profile = build_profile_urgent_normal(cfg, geom, opts)
%BUILD_PROFILE_URGENT_NORMAL Build per-user profile from a single group source.
%
% Source-of-truth policy:
% 1) Group labels come from geom.user_type (preferred) or opts.user_type.
% 2) Labels are normalized into profile.groups.urgent_idx/normal_idx.
% 3) Downstream code must use profile.groups.* only.

if nargin < 3 || isempty(opts)
    opts = struct();
end

K = cfg.num_users;

urgent_weight_row = get_opt(opts, 'urgent_weight_row', 1);
normal_weight_row = get_opt(opts, 'normal_weight_row', 2);
urgent_deadline = get_opt(opts, 'urgent_deadline', cfg.deadlines(1));
normal_deadline = get_opt(opts, 'normal_deadline', cfg.deadlines(min(2, end)));
dmax_all = get_opt(opts, 'dmax_all', cfg.dmax);
M_all = get_opt(opts, 'M_all', cfg.m_k);

[urgent_idx, normal_idx] = split_groups(K, geom, opts);

profile.M_k = as_kvec(M_all, K, 'M_all');

num_rows = size(cfg.weights, 1);
urgent_weight_row = min(max(round(urgent_weight_row), 1), num_rows);
normal_weight_row = min(max(round(normal_weight_row), 1), num_rows);

profile.weights = zeros(K, 2);
profile.weights(urgent_idx, :) = repmat(cfg.weights(urgent_weight_row, :), numel(urgent_idx), 1);
profile.weights(normal_idx, :) = repmat(cfg.weights(normal_weight_row, :), numel(normal_idx), 1);

profile.d_k = zeros(K, 1);
profile.d_k(urgent_idx) = urgent_deadline;
profile.d_k(normal_idx) = normal_deadline;

profile.dmax_k = as_kvec(dmax_all, K, 'dmax_all');

profile.groups.urgent_idx = urgent_idx(:);
profile.groups.normal_idx = normal_idx(:);
end

function [urgent_idx, normal_idx] = split_groups(K, geom, opts)
ut = strings(0, 1);

if nargin >= 2 && ~isempty(geom) && isstruct(geom) && isfield(geom, 'user_type') && ~isempty(geom.user_type)
    ut = to_strvec(geom.user_type);
elseif nargin >= 3 && isfield(opts, 'user_type') && ~isempty(opts.user_type)
    ut = to_strvec(opts.user_type);
end

if isempty(ut)
    error('build_profile_urgent_normal:missing_user_type', ...
        'Missing group source. Provide geom.user_type (or opts.user_type) to build profile.groups.');
end

n = min(numel(ut), K);
ut = lower(strtrim(ut(1:n)));
if n < K
    ut = [ut; repmat("normal", K - n, 1)];
end

urgent_idx = normalize_index_vector(ut == "urgent", K);
normal_idx = normalize_index_vector(ut == "normal", K);

normal_idx = setdiff(normal_idx, urgent_idx, 'stable');
covered = false(K, 1);
covered(urgent_idx) = true;
covered(normal_idx) = true;
missing = find(~covered);
normal_idx = [normal_idx; missing];

urgent_idx = sort(urgent_idx(:));
normal_idx = sort(normal_idx(:));
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

function v = get_opt(opts, field_name, default_v)
if isfield(opts, field_name) && ~isempty(opts.(field_name))
    v = opts.(field_name);
else
    v = default_v;
end
end
