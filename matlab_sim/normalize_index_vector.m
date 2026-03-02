function idx = normalize_index_vector(x, K)
%NORMALIZE_INDEX_VECTOR Normalize index input into a valid 1-based index vector.
%  Supports:
%   - logical mask (Kx1 or 1xK)
%   - numeric index vector (1-based)
%   - 0/1 numeric mask (treated as logical)
% Returns:
%   idx: column vector of unique indices in [1, K]

if nargin < 2 || isempty(K)
    % if K not provided, infer when possible
    if islogical(x)
        K = numel(x);
    else
        K = max([numel(x), max(x(:))]);
    end
end

if isempty(x)
    idx = zeros(0,1);
    return;
end

% Convert 0/1 numeric mask into logical mask
if isnumeric(x) && ~isscalar(x)
    ux = unique(x(~isnan(x)));
    if all(ismember(ux, [0 1]))
        x = logical(x);
    end
end

if islogical(x)
    x = x(:);
    if numel(x) ~= K
        % Try best effort: pad/trim to K
        x = x(1:min(end,K));
        if numel(x) < K
            x(end+1:K) = false;
        end
    end
    idx = find(x);
else
    idx = unique(round(double(x(:))));
    idx = idx(isfinite(idx));
    idx = idx(idx >= 1 & idx <= K);
end

idx = idx(:);
end