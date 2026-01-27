function assign = random_match(cfg, seed)
%RANDOM_MATCH Random RIS assignment with capacity.

if nargin < 2
    seed = 1;
end

rng(seed, 'twister');

assign = zeros(cfg.num_users, 1);
capacity = cfg.k0 * ones(cfg.num_ris, 1);

for k = 1:cfg.num_users
    options = [0; find(capacity > 0)];
    pick = options(randi(numel(options)));
    assign(k) = pick;
    if pick > 0
        capacity(pick) = capacity(pick) - 1;
    end
end
end
