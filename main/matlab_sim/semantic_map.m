function xi = semantic_map(gamma, M, mode, table_path, params)
%SEMANTIC_MAP Map SINR/SNR (linear) to semantic similarity xi in [0,1].
%
% Usage:
%   xi = semantic_map(gamma, M, 'proxy', '', params)
%   xi = semantic_map(gamma, M, 'table', 'semantic_tables/semantic_similarity_table.mat')
%   xi = semantic_map(gamma, M, 'table', 'semantic_tables/deepsc_table.csv')
%
% Inputs:
%   gamma      : linear SINR/SNR (scalar or array), NOT in dB
%   M          : semantic block size / symbol count (scalar or array, broadcastable with gamma)
%   mode       : 'proxy' | 'table' | 'deepsc_stub'
%   table_path : path to CSV or MAT file for mode='table'
%   params     : struct for proxy parameters / options
%
% Proxy default:
%   xi = (1-exp(-a*gamma)) * (1-exp(-b*M))
%   with a=0.6, b=0.4
%
% Table formats:
%   1) MAT must contain: snr_range (1xNs), n_sym (1xNm), sem_table (Nm x Ns)
%      where sem_table(i,j) = xi(n_sym(i), snr_range(j))
%   2) CSV long table with columns: snr_db, M, xi
%
% Notes:
%   - We interpret table x-axis as snr_db (in dB) computed from gamma: snr_db = 10*log10(gamma)
%   - For stability we clip queries to table bounds by default.

    if nargin < 3 || isempty(mode)
        mode = 'proxy';
    end
    if nargin < 4
        table_path = '';
    end
    if nargin < 5 || isempty(params)
        params = struct();
    end

    switch lower(mode)
        case 'proxy'
            xi = proxy_map(gamma, M, params);

        case 'table'
            xi = interp_table(gamma, M, table_path, params);

        case 'deepsc_stub'
            % Placeholder: default fallback to proxy
            xi = proxy_map(gamma, M, params);

        otherwise
            error('Unknown semantic mode: %s', mode);
    end
end

% ========================= helpers =========================

function xi = proxy_map(gamma, M, params)
    if ~isfield(params, 'a') || isempty(params.a)
        params.a = 0.6;
    end
    if ~isfield(params, 'b') || isempty(params.b)
        params.b = 0.4;
    end

    % numerical safety
    epsv = 1e-12;
    g = max(gamma, epsv);
    m = max(M, 0);

    xi = (1 - exp(-params.a .* g)) .* (1 - exp(-params.b .* m));
    xi = min(max(xi, 0), 1);
end

function xi = interp_table(gamma, M, table_path, params)
    persistent cache

    if isempty(table_path)
        error('Table path is required for semantic mode table.');
    end

    % options
    if ~isfield(params,'clip') || isempty(params.clip)
        params.clip = true; % recommended
    end
    epsv = 1e-12;

    % (re)load if cache miss or different file
    if isempty(cache) || ~isfield(cache, 'path') || ~strcmp(cache.path, table_path)
        [~,~,ext] = fileparts(table_path);

        if strcmpi(ext, '.mat')
            cache = build_interp_from_mat(table_path);
        else
            cache = build_interp_from_csv(table_path);
        end
    end

    % query
   snr_db = 10 * log10(max(gamma, 1e-12));  % gamma linear -> dB

if params.clip
    snr_db = min(max(snr_db, cache.snr_min), cache.snr_max);
    Mq     = min(max(M,      cache.m_min),   cache.m_max);
else
    Mq = M;
end

% --- Broadcast/align query sizes for griddedInterpolant ---
% cache.F is defined as F(M, snr_db)
if ~isequal(size(Mq), size(snr_db))
    if isscalar(Mq) && ~isscalar(snr_db)
        Mq = repmat(Mq, size(snr_db));
    elseif isscalar(snr_db) && ~isscalar(Mq)
        snr_db = repmat(snr_db, size(Mq));
    elseif numel(Mq) == numel(snr_db)
        % same number of elements but different shape (row vs col)
        Mq = reshape(Mq, size(snr_db));
    else
        error('semantic_map:SizeMismatch', ...
            'Query arrays must have same size (got M=%s, snr_db=%s).', mat2str(size(Mq)), mat2str(size(snr_db)));
    end
end

xi = cache.F(Mq, snr_db);
xi(isnan(xi)) = 0;
xi = min(max(xi, 0), 1);

end

function cache = build_interp_from_mat(table_path)
    S = load(table_path);

    if ~(isfield(S,'snr_range') && isfield(S,'n_sym') && isfield(S,'sem_table'))
        error('MAT file must contain variables: snr_range, n_sym, sem_table.');
    end

    snr_vals = S.snr_range(:);  % Ns x 1
    m_vals   = S.n_sym(:);      % Nm x 1
    xi_grid  = S.sem_table;     % Nm x Ns (rows=M, cols=SNR)

    if size(xi_grid,1) ~= numel(m_vals) || size(xi_grid,2) ~= numel(snr_vals)
        error('sem_table size must be [numel(n_sym) x numel(snr_range)].');
    end

    % ensure monotonically increasing axes (recommended)
    [snr_vals, snr_order] = sort(snr_vals, 'ascend');
    [m_vals,   m_order]   = sort(m_vals,   'ascend');
    xi_grid = xi_grid(m_order, snr_order);

    cache = struct();
    cache.path = table_path;
    cache.snr_vals = snr_vals;
    cache.m_vals = m_vals;
    cache.snr_min = snr_vals(1);
    cache.snr_max = snr_vals(end);
    cache.m_min   = m_vals(1);
    cache.m_max   = m_vals(end);

    % Interpolant over grid {M, SNRdB} with Z size [len(M) x len(SNR)]
    cache.F = griddedInterpolant({m_vals, snr_vals}, xi_grid, 'linear', 'nearest');
end

function cache = build_interp_from_csv(table_path)
    tbl = readtable(table_path);

    if ~all(ismember({'snr_db','M','xi'}, tbl.Properties.VariableNames))
        error('CSV must contain columns: snr_db, M, xi (long table format).');
    end

    snr_vals = unique(tbl.snr_db);
    m_vals   = unique(tbl.M);

    snr_vals = sort(snr_vals(:), 'ascend');
    m_vals   = sort(m_vals(:),   'ascend');

    % Build Z grid as [len(M) x len(SNR)] so that F(M, SNRdB) works
    xi_grid = nan(numel(m_vals), numel(snr_vals));

    for i = 1:height(tbl)
        s_idx = find(snr_vals == tbl.snr_db(i), 1);
        m_idx = find(m_vals   == tbl.M(i),      1);
        if ~isempty(s_idx) && ~isempty(m_idx)
            xi_grid(m_idx, s_idx) = tbl.xi(i); % row=M, col=SNR
        end
    end

    % Fill missing with 0 (conservative); you can change policy if needed
    xi_grid(isnan(xi_grid)) = 0;

    cache = struct();
    cache.path = table_path;
    cache.snr_vals = snr_vals;
    cache.m_vals = m_vals;
    cache.snr_min = snr_vals(1);
    cache.snr_max = snr_vals(end);
    cache.m_min   = m_vals(1);
    cache.m_max   = m_vals(end);

    cache.F = griddedInterpolant({m_vals, snr_vals}, xi_grid, 'linear', 'nearest');
end
