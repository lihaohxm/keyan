function [gamma, rates, sum_rate] = sinr_rate(cfg, h_eff, p_dbw)
%SINR_RATE Compute SINR and rates.

num_users = size(h_eff, 2);

p_watts = 10.^(p_dbw / 10);
p_k = p_watts / num_users;

w = zeros(cfg.nt, num_users);
for k = 1:num_users
    hk = h_eff(:, k);
    w(:, k) = hk / (norm(hk) + cfg.eps);
end

signal = zeros(num_users, 1);
interference = zeros(num_users, 1);

for k = 1:num_users
    hk = h_eff(:, k);
    for i = 1:num_users
        gain = abs(hk' * w(:, i)).^2;
        if i == k
            signal(k) = signal(k) + p_k * gain;
        else
            interference(k) = interference(k) + p_k * gain;
        end
    end
end

gamma = signal ./ (interference + cfg.noise_watts + cfg.eps);

rates = cfg.bandwidth * log2(1 + gamma);
sum_rate = sum(rates);
end
