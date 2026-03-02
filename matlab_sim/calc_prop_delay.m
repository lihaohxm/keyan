function prop_delay = calc_prop_delay(cfg, geom, assign, opts)
%CALC_PROP_DELAY Unified propagation-delay calculator.
%
% prop_delay(k):
%   assign(k)=0  -> direct link delay (mode-dependent)
%   assign(k)=l>0 -> factor * (|BS-RIS_l| + |RIS_l-UE_k|)

if nargin < 4 || isempty(opts)
    opts = struct();
end
if nargin < 3 || isempty(assign)
    if nargin >= 1 && isstruct(cfg) && isfield(cfg, 'num_users')
        assign = zeros(cfg.num_users, 1);
    else
        assign = zeros(0, 1);
    end
end

assign = assign(:);
K = numel(assign);
prop_delay = zeros(K, 1);

% factor priority: opts.factor > cfg.prop_delay_factor > 1e-5
factor = 1e-5;
if nargin >= 1 && isstruct(cfg) && isfield(cfg, 'prop_delay_factor') && ~isempty(cfg.prop_delay_factor)
    factor = cfg.prop_delay_factor;
end
if isfield(opts, 'factor') && ~isempty(opts.factor)
    factor = opts.factor;
end

direct_mode = 'zero';
if isfield(opts, 'direct_mode') && ~isempty(opts.direct_mode)
    direct_mode = lower(char(opts.direct_mode));
end

% Missing geometry/fields -> all zeros
if nargin < 2 || isempty(geom) || ~isstruct(geom) || ...
        ~isfield(geom, 'bs') || ~isfield(geom, 'ue') || ~isfield(geom, 'ris') || ...
        isempty(geom.bs) || isempty(geom.ue) || isempty(geom.ris)
    return;
end

bs = geom.bs;
if size(bs, 1) > 1, bs = bs(1, :); end
if numel(bs) >= 2, bs = bs(1:2); else, return; end

ue = geom.ue;
ris = geom.ris;
if size(ue, 1) < K
    return;
end

for k = 1:K
    if size(ue, 2) < 2, return; end
    ue_k = ue(k, 1:2);
    l = assign(k);

    if l == 0
        if strcmpi(direct_mode, 'bs_ue')
            prop_delay(k) = factor * norm(ue_k - bs);
        end
    elseif l > 0 && l <= size(ris, 1) && size(ris, 2) >= 2
        ris_l = ris(l, 1:2);
        prop_delay(k) = factor * (norm(ris_l - bs) + norm(ue_k - ris_l));
    end
end

end

