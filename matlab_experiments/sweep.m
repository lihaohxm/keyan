function result = sweep(varargin)
%SWEEP Sweep over power/M/k0.

addpath('matlab_sim');
addpath(fullfile('matlab_sim', 'matching'));
addpath('matlab_scripts');

cfg = config();

p = inputParser;
addParameter(p, 'seed', 1);
addParameter(p, 'mc', cfg.mc);
addParameter(p, 'semantic_mode', cfg.semantic_mode);
addParameter(p, 'table_path', cfg.semantic_table);
addParameter(p, 'x', 'p_dbw');
parse(p, varargin{:});

seed = p.Results.seed;
mc = p.Results.mc;
semantic_mode = p.Results.semantic_mode;
table_path = p.Results.table_path;
x_axis = p.Results.x;

switch x_axis
    case 'p_dbw'
        x_vals = cfg.p_dbw_list;
    case 'm_k'
        x_vals = [4 8 16 32];
    case 'k0'
        x_vals = [2 4 6];
    otherwise
        error('Unknown sweep axis.');
end

algorithms = {'random', 'norm', 'qoe', 'exhaustive', 'ga_placeholder'};

sum_rate = nan(numel(x_vals), numel(algorithms));
avg_qoe = nan(numel(x_vals), numel(algorithms));

for x = 1:numel(x_vals)
    cfg_local = cfg;
    if strcmp(x_axis, 'p_dbw')
        p_dbw = x_vals(x);
    elseif strcmp(x_axis, 'm_k')
        cfg_local.m_k = x_vals(x);
        p_dbw = cfg.p_dbw_list(1);
    else
        cfg_local.k0 = x_vals(x);
        p_dbw = cfg.p_dbw_list(1);
    end

    sum_rate_mc = zeros(mc, numel(algorithms));
    avg_qoe_mc = zeros(mc, numel(algorithms));

    for trial = 1:mc
        trial_seed = seed + trial - 1 + x * 1000;
        geom = geometry(cfg_local, trial_seed);
        ch = channel(cfg_local, geom, trial_seed);

        for a = 1:numel(algorithms)
            alg = algorithms{a};
            assign = pick_assignment(cfg_local, ch, alg, p_dbw, semantic_mode, table_path, cfg_local.weights(1, :), trial_seed);
            h_eff = effective_channel(cfg_local, ch, assign);
            [gamma, ~, sum_rate_mc(trial, a)] = sinr_rate(cfg_local, h_eff, p_dbw);
            xi = semantic_map(gamma, cfg_local.m_k, semantic_mode, table_path, struct('a', cfg_local.proxy_a, 'b', cfg_local.proxy_b));
            [avg_qoe_mc(trial, a), ~, ~] = qoe(cfg_local, gamma, cfg_local.m_k, xi, cfg_local.weights(1, :));
        end
    end

    sum_rate(x, :) = mean(sum_rate_mc, 1);
    avg_qoe(x, :) = mean(avg_qoe_mc, 1);
end

result.algorithms = algorithms;
result.x_axis = x_axis;
result.x_vals = x_vals;
result.sum_rate = sum_rate;
result.avg_qoe = avg_qoe;

run_id = save_results('sweep', result, x_vals);
result.run_id = run_id;

fprintf('Sweep complete. Results saved with run_id: %s\n', run_id);
end

function assign = pick_assignment(cfg, ch, alg, p_dbw, semantic_mode, table_path, weights, seed)
    switch alg
        case 'random'
            assign = random_match(cfg, seed);
        case 'norm'
            assign = norm_based(cfg, ch);
        case 'qoe'
            assign = qoe_aware(cfg, ch, p_dbw, semantic_mode, table_path, weights);
        case 'exhaustive'
            [assign, info] = exhaustive(cfg, ch, p_dbw);
            if info.skipped
                assign = norm_based(cfg, ch);
            end
        case 'ga_placeholder'
            assign = ga_placeholder(cfg);
        otherwise
            assign = norm_based(cfg, ch);
    end
end
