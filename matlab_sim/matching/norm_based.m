function assign = norm_based(cfg, ch)
%NORM_BASED Greedy assignment based on link norm.

num_users = cfg.num_users;
num_ris = cfg.num_ris;

strength = zeros(num_users, num_ris + 1);

for k = 1:num_users
    strength(k, 1) = norm(ch.h_d(:, k));
    for l = 1:num_ris
        theta = ch.theta(:, l);
        refl = ch.G(:, :, l) * (theta .* ch.H_ris(:, k, l));
        strength(k, l + 1) = norm(refl);
    end
end

capacity = cfg.k0 * ones(num_ris, 1);
assign = zeros(num_users, 1);

[~, order] = sort(max(strength, [], 2), 'descend');

for idx = 1:num_users
    k = order(idx);
    [~, choices] = sort(strength(k, :), 'descend');
    assigned = false;
    for c = 1:numel(choices)
        l = choices(c) - 1;
        if l == 0
            assign(k) = 0;
            assigned = true;
            break;
        elseif capacity(l) > 0
            assign(k) = l;
            capacity(l) = capacity(l) - 1;
            assigned = true;
            break;
        end
    end
    if ~assigned
        assign(k) = 0;
    end
end
end
