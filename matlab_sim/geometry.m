function geom = geometry(cfg, seed)
%GEOMETRY Generate BS, UE, RIS positions.

if nargin < 2
    seed = 1;
end

rng(seed, 'twister');

bs = cfg.bs_positions;

num_users = cfg.num_users;
num_ris = cfg.num_ris;

ue = zeros(num_users, 2);
ris = zeros(num_ris, 2);

idx = 1;
ris_idx = 1;
for cell = 1:cfg.num_cells
    bs_pos = bs(cell, :);

    for u = 1:cfg.users_per_cell
        r = 80 + 80 * rand();
        ang = 2 * pi * rand();
        ue(idx, :) = bs_pos + r * [cos(ang) sin(ang)];
        idx = idx + 1;
    end

    for r = 1:cfg.ris_per_cell
        r_dist = 50 + 30 * rand();
        r_ang = 2 * pi * rand();
        ris(ris_idx, :) = bs_pos + r_dist * [cos(r_ang) sin(r_ang)];
        ris_idx = ris_idx + 1;
    end
end

geom.bs = bs;
geom.ue = ue;
geom.ris = ris;
end
