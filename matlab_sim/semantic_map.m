function xi = semantic_map(gamma, M, mode, table_path, params)
%SEMANTIC_MAP Map SINR/SNR to semantic similarity.

if nargin < 3 || isempty(mode)
    mode = 'proxy';
end
if nargin < 4
    table_path = '';
end
if nargin < 5
    params = struct();
end

switch lower(mode)
    case 'proxy'
        if ~isfield(params, 'a')
            params.a = 0.6;
        end
        if ~isfield(params, 'b')
            params.b = 0.4;
        end
        xi = (1 - exp(-params.a .* gamma)) .* (1 - exp(-params.b .* M));
        xi = min(max(xi, 0), 1);
    case 'table'
        xi = interp_table(gamma, M, table_path);
    case 'deepsc_stub'
        xi = semantic_map(gamma, M, 'proxy', '', params);
    otherwise
        error('Unknown semantic mode: %s', mode);
end
end

function xi = interp_table(gamma, M, table_path)
    persistent cache

    if isempty(table_path)
        error('Table path is required for semantic mode table.');
    end

    if isempty(cache) || ~isfield(cache, 'path') || ~strcmp(cache.path, table_path)
        tbl = readtable(table_path);
        if ~all(ismember({'snr_db','M','xi'}, tbl.Properties.VariableNames))
            error('CSV must contain snr_db, M, xi columns.');
        end

        snr_vals = unique(tbl.snr_db);
        m_vals = unique(tbl.M);
        xi_grid = nan(numel(snr_vals), numel(m_vals));

        for i = 1:height(tbl)
            s_idx = find(snr_vals == tbl.snr_db(i), 1);
            m_idx = find(m_vals == tbl.M(i), 1);
            xi_grid(s_idx, m_idx) = tbl.xi(i);
        end

        fill_mask = isnan(xi_grid);
        if any(fill_mask(:))
            xi_grid(fill_mask) = 0;
        end

        cache.path = table_path;
        cache.snr_vals = snr_vals;
        cache.m_vals = m_vals;
        cache.interp = griddedInterpolant({snr_vals, m_vals}, xi_grid, 'linear', 'nearest');
    end

    snr_db = 10 * log10(gamma + 1e-9);
    snr_db = min(max(snr_db, cache.snr_vals(1)), cache.snr_vals(end));
    M = min(max(M, cache.m_vals(1)), cache.m_vals(end));

    xi = cache.interp(snr_db, M);
    xi = min(max(xi, 0), 1);
end
