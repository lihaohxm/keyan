function geom = geometry(cfg, seed)
%GEOMETRY Generate BS, UE, RIS positions.
%
% RIS at varied distances to create delay-SINR trade-offs.

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

    % Users: 60-150m from BS
    for u = 1:cfg.users_per_cell
        r = 60 + 90 * rand();
        ang = 2 * pi * rand();
        ue(idx, :) = bs_pos + r * [cos(ang) sin(ang)];
        idx = idx + 1;
    end

    % RIS: half near (20-40m), half far (70-110m)
    for r = 1:cfg.ris_per_cell
        if r <= cfg.ris_per_cell / 2
            r_dist = 20 + 20 * rand();
        else
            r_dist = 70 + 40 * rand();
        end
        r_ang = 2 * pi * (r - 1) / cfg.ris_per_cell + 0.2 * rand();
        ris(ris_idx, :) = bs_pos + r_dist * [cos(r_ang) sin(r_ang)];
        ris_idx = ris_idx + 1;
    end
end

geom.bs = bs;
geom.ue = ue;
geom.ris = ris;
geom.ris_dist_from_bs = sqrt(sum((ris - bs).^2, 2));
end
