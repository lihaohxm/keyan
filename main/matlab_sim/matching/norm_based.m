function assign = norm_based(cfg, ch)
%NORM_BASED Greedy assignment based on effective channel norm.
% Uses the same physical model knob cfg.ris_gain as effective_channel.m.

num_users = cfg.num_users;
num_ris = cfg.num_ris;

if ~isfield(cfg, 'ris_gain') || isempty(cfg.ris_gain)
    cfg.ris_gain = 1;
end

strength = zeros(num_users, num_ris + 1);

for k = 1:num_users
    hdk = ch.h_d(:, k);
    strength(k, 1) = norm(hdk);

    for l = 1:num_ris
        theta = ch.theta(:, l);
        refl = ch.G(:, :, l) * (theta .* ch.H_ris(:, k, l));
        h_eff_k = hdk + cfg.ris_gain * refl;
        strength(k, l + 1) = norm(h_eff_k);
    end
end

capacity = cfg.k0 * ones(num_ris, 1);
assign = zeros(num_users, 1);

% users with larger best gain go first
[~, order] = sort(max(strength, [], 2), 'descend');

for idx = 1:num_users
    k = order(idx);

    % pick strongest option first
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
