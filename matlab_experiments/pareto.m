function result = pareto(varargin)
%PARETO Sweep QoE weights and produce Pareto points.

addpath('matlab_sim');
addpath(fullfile('matlab_sim', 'matching'));
addpath('matlab_scripts');

cfg = config();

p = inputParser;
addParameter(p, 'seed', 1);
addParameter(p, 'mc', cfg.mc);
addParameter(p, 'semantic_mode', cfg.semantic_mode);
addParameter(p, 'table_path', cfg.semantic_table);
parse(p, varargin{:});

seed = p.Results.seed;
mc = p.Results.mc;
semantic_mode = p.Results.semantic_mode;
table_path = p.Results.table_path;

algorithms = {'random', 'norm', 'qoe', 'exhaustive', 'ga_placeholder'};

num_weights = size(cfg.weights, 1);

sum_rate = nan(num_weights, numel(algorithms));
avg_qoe  = nan(num_weights, numel(algorithms));

deadline_vio = nan(num_weights, numel(algorithms));
semantic_vio = nan(num_weights, numel(algorithms));

p_dbw = cfg.p_dbw_list(1);

for w = 1:num_weights
    weights = cfg.weights(w, :);

    sum_rate_mc = zeros(mc, numel(algorithms));
    avg_qoe_mc  = zeros(mc, numel(algorithms));

    deadline_vio_mc = zeros(mc, numel(algorithms));
    semantic_vio_mc = zeros(mc, numel(algorithms));

    for trial = 1:mc
        trial_seed = seed + trial - 1 + w * 2000;
        geom = geometry(cfg, trial_seed);
        ch = channel(cfg, geom, trial_seed);

        for a = 1:numel(algorithms)
            alg = algorithms{a};

            assign = pick_assignment(cfg, ch, alg, p_dbw, semantic_mode, table_path, weights, trial_seed);
            out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights, semantic_mode, table_path);
            sum_rate_mc(trial, a) = out.sum_rate_bps;
            avg_qoe_mc(trial, a) = out.avg_qoe;
            deadline_vio_mc(trial, a) = out.delay_vio_rate_all;
            semantic_vio_mc(trial, a) = out.semantic_vio_rate_all;
        end
    end

    sum_rate(w, :) = mean(sum_rate_mc, 1);
    avg_qoe(w, :)  = mean(avg_qoe_mc, 1);

    deadline_vio(w, :) = mean(deadline_vio_mc, 1);
    semantic_vio(w, :) = mean(semantic_vio_mc, 1);
end

result.algorithms = algorithms;
result.weights = cfg.weights;

result.sum_rate = sum_rate;
result.avg_qoe  = avg_qoe;

result.deadline_vio = deadline_vio;
result.semantic_vio = semantic_vio;

run_id = save_results('pareto', result, (1:num_weights));
result.run_id = run_id;

fprintf('Pareto complete. Results saved with run_id: %s\n', run_id);
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

function out = evaluate_assignment_rsma(cfg, ch, geom, assign, p_dbw, weights, semantic_mode, table_path)
    profile = build_profile_urgent_normal(cfg, geom, struct());
    profile.weights = repmat(weights(:).', cfg.num_users, 1);
    h_eff = effective_channel(cfg, ch, assign);
    [V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 5);
    sol = struct('assign', assign, 'theta_all', ch.theta, 'V', V);
    eval_opts = struct('semantic_mode', semantic_mode, 'table_path', table_path);
    out = evaluate_system_rsma(cfg, ch, geom, sol, profile, eval_opts);
end
