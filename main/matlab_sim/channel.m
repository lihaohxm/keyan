function ch = channel(cfg, geom, seed)
%CHANNEL Generate large/small-scale channels.

if nargin < 3
    seed = 1;
end

rng(seed, 'twister');

num_users = cfg.num_users;
num_ris = cfg.num_ris;

h_d = zeros(cfg.nt, num_users);
G = zeros(cfg.nt, cfg.n_ris, num_ris);
H_ris = zeros(cfg.n_ris, num_users, num_ris);

for k = 1:num_users
    d = norm(geom.ue(k, :) - geom.bs(1, :)) + cfg.eps;
    pathloss = d.^(-cfg.pathloss_exp);
    h_d(:, k) = sqrt(pathloss/2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
end

for l = 1:num_ris
    d_br = norm(geom.ris(l, :) - geom.bs(1, :)) + cfg.eps;
    pl_br = d_br.^(-cfg.pathloss_exp);
    G(:, :, l) = sqrt(pl_br/2) * (randn(cfg.nt, cfg.n_ris) + 1j * randn(cfg.nt, cfg.n_ris));

    for k = 1:num_users
        d_ru = norm(geom.ue(k, :) - geom.ris(l, :)) + cfg.eps;
        pl_ru = d_ru.^(-cfg.pathloss_exp);
        H_ris(:, k, l) = sqrt(pl_ru/2) * (randn(cfg.n_ris, 1) + 1j * randn(cfg.n_ris, 1));
    end
end

theta = exp(1j * 2 * pi * rand(cfg.n_ris, num_ris));

ch.h_d = h_d;
ch.G = G;
ch.H_ris = H_ris;
ch.theta = theta;
end
