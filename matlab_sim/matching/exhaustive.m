function [assign, info] = exhaustive(cfg, ch, p_dbw)
%EXHAUSTIVE Exhaustive assignment for small sizes.

info.skipped = false;
info.reason = '';

if cfg.num_users > 6 || cfg.num_ris > 4
    info.skipped = true;
    info.reason = 'Problem size too large for exhaustive search.';
    assign = zeros(cfg.num_users, 1);
    return;
end

num_users = cfg.num_users;
num_ris = cfg.num_ris;

best_sum_rate = -inf;
assign = zeros(num_users, 1);

options = 0:num_ris;

combos = cell(1, num_users);
for k = 1:num_users
    combos{k} = options;
end

idxs = cell(1, num_users);
[idxs{:}] = ndgrid(combos{:});

all_assign = zeros(numel(idxs{1}), num_users);
for k = 1:num_users
    all_assign(:, k) = idxs{k}(:);
end

for row = 1:size(all_assign, 1)
    candidate = all_assign(row, :).';
    if ~capacity_ok(candidate, cfg)
        continue;
    end
    h_eff = effective_channel(cfg, ch, candidate);
    [~, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
    if sum_rate > best_sum_rate
        best_sum_rate = sum_rate;
        assign = candidate;
    end
end
end

function ok = capacity_ok(assign, cfg)
    ok = true;
    for l = 1:cfg.num_ris
        if sum(assign == l) > cfg.k0
            ok = false;
            return;
        end
    end
end
