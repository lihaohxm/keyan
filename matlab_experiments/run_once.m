function result = run_once(varargin)
%RUN_ONCE Run a single simulation with fixed parameters.

addpath('matlab_sim');
addpath(fullfile('matlab_sim', 'matching'));
addpath('matlab_scripts');

cfg = config();

p = inputParser;
addParameter(p, 'seed', 1);
addParameter(p, 'mc', cfg.mc);
addParameter(p, 'p_dbw', cfg.p_dbw_list(1));
addParameter(p, 'semantic_mode', cfg.semantic_mode);
addParameter(p, 'table_path', cfg.semantic_table);
addParameter(p, 'weights', cfg.weights(1, :));
parse(p, varargin{:});

seed = p.Results.seed;
mc = p.Results.mc;
p_dbw = p.Results.p_dbw;
semantic_mode = p.Results.semantic_mode;
table_path = p.Results.table_path;
weights = p.Results.weights;

cfg.semantic_mode = semantic_mode;

algorithms = {'random', 'norm', 'qoe', 'exhaustive', 'ga_placeholder'};

sum_rate = zeros(mc, numel(algorithms));
avg_qoe = zeros(mc, numel(algorithms));

for trial = 1:mc
    trial_seed = seed + trial - 1;
    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);

    for a = 1:numel(algorithms)
        alg = algorithms{a};
        assign = pick_assignment(cfg, ch, alg, p_dbw, semantic_mode, table_path, weights, trial_seed);
        h_eff = effective_channel(cfg, ch, assign);
        [gamma, ~, sum_rate(trial, a)] = sinr_rate(cfg, h_eff, p_dbw);
        xi = semantic_map(gamma, cfg.m_k, semantic_mode, table_path, struct('a', cfg.proxy_a, 'b', cfg.proxy_b));
        [avg_qoe(trial, a), ~, ~] = qoe(cfg, gamma, cfg.m_k, xi, weights);
    end
end

result.algorithms = algorithms;
result.sum_rate = mean(sum_rate, 1);
result.avg_qoe = mean(avg_qoe, 1);
result.sum_rate_std = std(sum_rate, 0, 1);
result.avg_qoe_std = std(avg_qoe, 0, 1);
result.seed = seed;
result.mc = mc;
result.p_dbw = p_dbw;
result.semantic_mode = semantic_mode;
result.table_path = table_path;

run_id = save_results('run_once', result, p_dbw);
result.run_id = run_id;

fprintf('Run complete. Results saved with run_id: %s\n', run_id);
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
