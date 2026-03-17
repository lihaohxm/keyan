function geom = geometry(cfg, seed)
%GEOMETRY Scenario geometry generator.
% Geometry is task-agnostic: urgency is assigned later from task pressure
% and a pre-scheduling channel-quality proxy.

if nargin < 2
    seed = 1;
end

rng(seed, 'twister');

bs = cfg.bs_positions;
if numel(bs) < 2
    error('cfg.bs_positions must contain at least two coordinates.');
end
bs = bs(1:2);

num_users = cfg.num_users;
num_ris = cfg.num_ris;

user_r_min = get_field(cfg, 'user_radius_min', 30);
user_r_max = get_field(cfg, 'user_radius_max', 180);
ris_r_min = get_field(cfg, 'ris_radius_min', 70);
ris_r_max = get_field(cfg, 'ris_radius_max', 110);

ue = sample_annulus_points(bs, num_users, user_r_min, user_r_max);
ris = sample_annulus_points(bs, num_ris, ris_r_min, ris_r_max);

geom.bs = bs;
geom.ue = ue;
geom.ris = ris;
geom.seed = seed;
geom.user_dist_from_bs = sqrt(sum((ue - bs).^2, 2));
geom.ris_dist_from_bs = sqrt(sum((ris - bs).^2, 2));
geom.user_angle_from_bs = atan2(ue(:, 2) - bs(2), ue(:, 1) - bs(1));
geom.ris_angle_from_bs = atan2(ris(:, 2) - bs(2), ris(:, 1) - bs(1));
geom.user_type = repmat("generic", num_users, 1);
end

function pts = sample_annulus_points(center, count, r_min, r_max)
pts = zeros(count, 2);
for idx = 1:count
    radius = r_min + (r_max - r_min) * rand();
    angle = 2 * pi * rand();
    pts(idx, :) = center + radius * [cos(angle), sin(angle)];
end
end

function v = get_field(s, f, d)
if isfield(s, f)
    v = s.(f);
else
    v = d;
end
end
