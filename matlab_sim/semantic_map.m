function xi = semantic_map(gamma, M, mode, table_path, params)
%SEMANTIC_MAP Map SINR (linear) to semantic similarity xi in [0,1].
%
% Proxy: xi = (1-exp(-a*gamma)) * (1-exp(-b*M))

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
        otherwise
            xi = proxy_map(gamma, M, params);
    end
end

function xi = proxy_map(gamma, M, params)
    if ~isfield(params, 'a') || isempty(params.a)
        params.a = 0.6;
    end
    if ~isfield(params, 'b') || isempty(params.b)
        params.b = 0.4;
    end

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

    if ~isfield(params,'clip') || isempty(params.clip)
        params.clip = true;
    end

    if isempty(cache) || ~isfield(cache, 'path') || ~strcmp(cache.path, table_path)
        [~,~,ext] = fileparts(table_path);
        if strcmpi(ext, '.mat')
            S = load(table_path);
            snr_vals = S.snr_range(:);
            m_vals = S.n_sym(:);
            xi_grid = S.sem_table;
            [snr_vals, snr_order] = sort(snr_vals, 'ascend');
            [m_vals, m_order] = sort(m_vals, 'ascend');
            xi_grid = xi_grid(m_order, snr_order);
        else
            tbl = readtable(table_path);
            snr_vals = unique(tbl.snr_db);
            m_vals = unique(tbl.M);
            snr_vals = sort(snr_vals(:), 'ascend');
            m_vals = sort(m_vals(:), 'ascend');
            xi_grid = nan(numel(m_vals), numel(snr_vals));
            for i = 1:height(tbl)
                s_idx = find(snr_vals == tbl.snr_db(i), 1);
                m_idx = find(m_vals == tbl.M(i), 1);
                if ~isempty(s_idx) && ~isempty(m_idx)
                    xi_grid(m_idx, s_idx) = tbl.xi(i);
                end
            end
            xi_grid(isnan(xi_grid)) = 0;
        end
        cache.path = table_path;
        cache.snr_vals = snr_vals;
        cache.m_vals = m_vals;
        cache.snr_min = snr_vals(1);
        cache.snr_max = snr_vals(end);
        cache.m_min = m_vals(1);
        cache.m_max = m_vals(end);
        cache.F = griddedInterpolant({m_vals, snr_vals}, xi_grid, 'linear', 'nearest');
    end

    snr_db = 10 * log10(max(gamma, 1e-12));

    if params.clip
        snr_db = min(max(snr_db, cache.snr_min), cache.snr_max);
        Mq = min(max(M, cache.m_min), cache.m_max);
    else
        Mq = M;
    end

    if ~isequal(size(Mq), size(snr_db))
        if isscalar(Mq) && ~isscalar(snr_db)
            Mq = repmat(Mq, size(snr_db));
        elseif isscalar(snr_db) && ~isscalar(Mq)
            snr_db = repmat(snr_db, size(Mq));
        elseif numel(Mq) == numel(snr_db)
            Mq = reshape(Mq, size(snr_db));
        end
    end

    xi = cache.F(Mq, snr_db);
    xi(isnan(xi)) = 0;
    xi = min(max(xi, 0), 1);
end
