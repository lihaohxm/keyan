function h_eff = effective_channel(cfg, ch, assign)
%EFFECTIVE_CHANNEL Compute effective channel with assignment.

num_users = cfg.num_users;

h_eff = ch.h_d;

for k = 1:num_users
    l = assign(k);
    if l > 0
        theta = ch.theta(:, l);
        h_ris = ch.H_ris(:, k, l);
        h_eff(:, k) = h_eff(:, k) + ch.G(:, :, l) * (theta .* h_ris);
    end
end
end
