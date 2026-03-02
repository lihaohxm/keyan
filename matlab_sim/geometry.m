function geom = geometry(cfg, seed)
%GEOMETRY Scenario geometry generator.
%
% Group source policy:
% - geometry only tags user types in geom.user_type
% - profile construction must normalize/use geom.user_type and store profile.groups.*

if nargin < 2
    seed = 1;
end

rng(seed, 'twister');

bs = cfg.bs_positions;
num_users = cfg.num_users;
num_ris = cfg.num_ris;
num_urgent = min(num_users, max(0, round(get_field(cfg, 'num_urgent', 4))));
num_normal = max(0, num_users - num_urgent);

ue = zeros(num_users, 2);
ris = zeros(num_ris, 2);

% Urgent users: farther and concentrated sector.
urgent_sector = [0, pi/4];
for k = 1:num_urgent
    r = 120 + 60 * rand();
    ang = urgent_sector(1) + (urgent_sector(2) - urgent_sector(1)) * rand();
    ue(k, :) = bs + r * [cos(ang) sin(ang)];
end

% Normal users: closer and isotropic.
for k = (num_urgent + 1):num_users
    r = 30 + 40 * rand();
    ang = 2 * pi * rand();
    ue(k, :) = bs + r * [cos(ang) sin(ang)];
end

% RIS deployment towards urgent sector.
for l = 1:num_ris
    r_dist = 80 + 20 * rand();
    ang = urgent_sector(1) + (urgent_sector(2) - urgent_sector(1)) * rand();
    ris(l, :) = bs + r_dist * [cos(ang) sin(ang)];
end

geom.bs = bs;
geom.ue = ue;
geom.ris = ris;
geom.ris_dist_from_bs = sqrt(sum((ris - bs).^2, 2));
geom.user_type = [repmat("urgent", num_urgent, 1); repmat("normal", num_normal, 1)];
end

function v = get_field(s, f, d)
if isfield(s, f)
    v = s.(f);
else
    v = d;
end
end
