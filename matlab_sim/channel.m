function ch = channel(cfg, geom, seed)
%CHANNEL Channel generation from geometry only.

if nargin < 3
    seed = 1;
end

rng(seed, 'twister');

num_users = cfg.num_users;
num_ris = cfg.num_ris;

h_d = zeros(cfg.nt, num_users);
G = zeros(cfg.nt, cfg.n_ris, num_ris);
H_ris = zeros(cfg.n_ris, num_users, num_ris);

alpha_direct = get_field(cfg, 'pathloss_exp_direct', get_field(cfg, 'pathloss_exp', 3.2));
alpha_br = get_field(cfg, 'pathloss_exp_br', 2.2);
alpha_ru = get_field(cfg, 'pathloss_exp_ru', 2.2);
blockage_loss_direct = get_field(cfg, 'blockage_loss_direct', 1.0);
blockage_loss_ru = get_field(cfg, 'blockage_loss_ru', 1.0);

bs_pos = geom.bs;
if numel(bs_pos) >= 3
    bs_pos = bs_pos(1:2);
end

for k = 1:num_users
    ue_pos = geom.ue(k, :);
    if numel(ue_pos) >= 3
        ue_pos = ue_pos(1:2);
    end
    d = norm(ue_pos - bs_pos) + cfg.eps;
    pathloss = d.^(-alpha_direct) * blockage_loss_direct;
    h_d(:, k) = sqrt(pathloss / 2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
end

K_br = get_field(cfg, 'rician_K_br', 10);
MAX_L = max(256, cfg.n_ris);

for l = 1:num_ris
    ris_pos = geom.ris(l, :);
    if numel(ris_pos) >= 3
        ris_pos = ris_pos(1:2);
    end

    d_br = norm(ris_pos - bs_pos) + cfg.eps;
    pl_br = d_br.^(-alpha_br);

    angle_br = atan2(ris_pos(2) - bs_pos(2), ris_pos(1) - bs_pos(1));
    a_bs = exp(1j * pi * (0:cfg.nt-1).' * sin(angle_br)) / sqrt(cfg.nt);
    a_ris = exp(1j * pi * (0:cfg.n_ris-1).' * sin(angle_br)) / sqrt(cfg.n_ris);
    G_los = a_bs * a_ris';

    G_nlos_pool = (randn(cfg.nt, MAX_L) + 1j * randn(cfg.nt, MAX_L)) / sqrt(2);
    G_nlos = G_nlos_pool(:, 1:cfg.n_ris);
    G(:, :, l) = sqrt(pl_br) * (sqrt(K_br / (K_br + 1)) * G_los + sqrt(1 / (K_br + 1)) * G_nlos);

    for k = 1:num_users
        ue_pos = geom.ue(k, :);
        if numel(ue_pos) >= 3
            ue_pos = ue_pos(1:2);
        end
        d_ru = norm(ue_pos - ris_pos) + cfg.eps;
        pl_ru = d_ru.^(-alpha_ru) * blockage_loss_ru;
        H_ris_pool = sqrt(pl_ru / 2) * (randn(MAX_L, 1) + 1j * randn(MAX_L, 1));
        H_ris(:, k, l) = H_ris_pool(1:cfg.n_ris);
    end
end

theta_pool = exp(1j * 2 * pi * rand(MAX_L, num_ris));
theta = theta_pool(1:cfg.n_ris, :);

ch.h_d = h_d;
ch.G = G;
ch.H_ris = H_ris;
ch.theta = theta;
end

function v = get_field(s, f, d)
if isfield(s, f)
    v = s.(f);
else
    v = d;
end
end
