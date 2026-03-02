function ch = channel(cfg, geom, seed)
%CHANNEL Channel generation with group labels from geom.user_type.

if nargin < 3
    seed = 1;
end

rng(seed, 'twister');

num_users = cfg.num_users;
num_ris = cfg.num_ris;

h_d = zeros(cfg.nt, num_users);
G = zeros(cfg.nt, cfg.n_ris, num_ris);
H_ris = zeros(cfg.n_ris, num_users, num_ris);

alpha_direct_urgent = get_field(cfg, 'pathloss_exp_direct_urgent', 4.2);
alpha_direct_normal = get_field(cfg, 'pathloss_exp_direct_normal', 3.0);

alpha_br = get_field(cfg, 'pathloss_exp_br', 2.2);
alpha_ru_urgent = get_field(cfg, 'pathloss_exp_ru_urgent', 2.6);
alpha_ru_normal = get_field(cfg, 'pathloss_exp_ru_normal', 2.2);

blockage_loss_urgent = get_field(cfg, 'blockage_loss_urgent', 10^(-15/10));
blockage_loss_normal = get_field(cfg, 'blockage_loss_normal', 1.0);

is_urgent = get_urgent_mask_from_geom(geom, num_users);

bs_pos = geom.bs;
if numel(bs_pos) == 3, bs_pos = bs_pos(1:2); end

for k = 1:num_users
    ue_pos = geom.ue(k, :);
    if numel(ue_pos) == 3, ue_pos = ue_pos(1:2); end
    d = norm(ue_pos - bs_pos) + cfg.eps;

    if is_urgent(k)
        alpha = alpha_direct_urgent;
        blk = blockage_loss_urgent;
    else
        alpha = alpha_direct_normal;
        blk = blockage_loss_normal;
    end

    pathloss = d.^(-alpha) * blk;
    h_d(:, k) = sqrt(pathloss/2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
end

for l = 1:num_ris
    ris_pos = geom.ris(l, :);
    if numel(ris_pos) == 3, ris_pos = ris_pos(1:2); end

    d_br = norm(ris_pos - bs_pos) + cfg.eps;
    pl_br = d_br.^(-alpha_br);
    G(:, :, l) = sqrt(pl_br/2) * (randn(cfg.nt, cfg.n_ris) + 1j * randn(cfg.nt, cfg.n_ris));

    for k = 1:num_users
        ue_pos = geom.ue(k, :);
        if numel(ue_pos) == 3, ue_pos = ue_pos(1:2); end
        d_ru = norm(ue_pos - ris_pos) + cfg.eps;

        if is_urgent(k)
            alpha_ru = alpha_ru_urgent;
            blk = blockage_loss_urgent;
        else
            alpha_ru = alpha_ru_normal;
            blk = blockage_loss_normal;
        end

        pl_ru = d_ru.^(-alpha_ru) * blk;
        H_ris(:, k, l) = sqrt(pl_ru/2) * (randn(cfg.n_ris, 1) + 1j * randn(cfg.n_ris, 1));
    end
end

theta = exp(1j * 2 * pi * rand(cfg.n_ris, num_ris));

ch.h_d = h_d;
ch.G = G;
ch.H_ris = H_ris;
ch.theta = theta;
end

function mask = get_urgent_mask_from_geom(geom, K)
mask = false(K, 1);
if ~isstruct(geom) || ~isfield(geom, 'user_type') || isempty(geom.user_type)
    return;
end
if isstring(geom.user_type)
    ut = geom.user_type(:);
elseif iscell(geom.user_type)
    ut = string(geom.user_type(:));
else
    return;
end
n = min(K, numel(ut));
mask(1:n) = lower(strtrim(ut(1:n))) == "urgent";
end

function v = get_field(s, f, d)
if isfield(s, f), v = s.(f); else, v = d; end
end
