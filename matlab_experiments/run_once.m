function result = run_once(varargin)
%RUN_ONCE Run a single simulation with fixed parameters.
%
% Example:
%   run_once('mc',1,'dmax',0.95,'beta_s',0.01);

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

% allow overriding QoE-related knobs without editing config.m
addParameter(p, 'dmax', cfg.dmax);
addParameter(p, 'beta_s', cfg.beta_s);
addParameter(p, 'beta_d', cfg.beta_d);
addParameter(p, 'b_s', cfg.b_s);
addParameter(p, 'b_d', cfg.b_d);

parse(p, varargin{:});

seed = p.Results.seed;
mc = p.Results.mc;
p_dbw = p.Results.p_dbw;
semantic_mode = p.Results.semantic_mode;
table_path = p.Results.table_path;
weights = p.Results.weights;

% Apply overrides (only for this run)
cfg.semantic_mode = semantic_mode;
cfg.dmax   = p.Results.dmax;
cfg.beta_s = p.Results.beta_s;
cfg.beta_d = p.Results.beta_d;
cfg.b_s    = p.Results.b_s;
cfg.b_d    = p.Results.b_d;

algorithms = {'random', 'norm', 'qoe', 'exhaustive', 'ga_placeholder'};

sum_rate = zeros(mc, numel(algorithms));
avg_qoe  = zeros(mc, numel(algorithms));

deadline_vio = zeros(mc, numel(algorithms));
semantic_vio = zeros(mc, numel(algorithms));

% D stats
D_mean = zeros(mc, numel(algorithms));
D_p10  = zeros(mc, numel(algorithms));
D_p50  = zeros(mc, numel(algorithms));
D_p90  = zeros(mc, numel(algorithms));

% assignment fingerprint (string)
assign_fp = strings(mc, numel(algorithms));

fprintf('Overrides: dmax=%.4f, beta_s=%.4g, beta_d=%.4g, b_s=%.4g, b_d=%.4g\n', ...
    cfg.dmax, cfg.beta_s, cfg.beta_d, cfg.b_s, cfg.b_d);

for trial = 1:mc
    trial_seed = seed + trial - 1;
    geom = geometry(cfg, trial_seed);
    ch = channel(cfg, geom, trial_seed);

    for a = 1:numel(algorithms)
        alg = algorithms{a};

        assign = pick_assignment(cfg, ch, alg, p_dbw, semantic_mode, table_path, weights, trial_seed);

        % ---- assignment fingerprint (quick & stable)
        assign_fp(trial, a) = assignment_fingerprint(assign);

        h_eff = effective_channel(cfg, ch, assign);

        [gamma, ~, sum_rate(trial, a)] = sinr_rate(cfg, h_eff, p_dbw);

        xi = semantic_map(gamma, cfg.m_k, semantic_mode, table_path, ...
            struct('a', cfg.proxy_a, 'b', cfg.proxy_b));

        D = 1 - xi;
        D_mean(trial, a) = mean(D);
        D_p10(trial, a)  = prctile(D, 10);
        D_p50(trial, a)  = prctile(D, 50);
        D_p90(trial, a)  = prctile(D, 90);

        [avg_qoe(trial, a), ~, meta] = qoe(cfg, gamma, cfg.m_k, xi, weights);

        deadline_vio(trial, a) = 1 - meta.deadline_ok;
        semantic_vio(trial, a) = 1 - meta.semantic_ok;

        fprintf([ ...
            '%s: fp=%s, mean(g)=%.4f, std(g)=%.4f, min(g)=%.4f, max(g)=%.4f, ' ...
            'mean(xi)=%.4f, D(mean/p10/p50/p90)=(%.4f/%.4f/%.4f/%.4f), ' ...
            'avgQ=%.6f, dv=%.4f, sv=%.4f\n'], ...
            alg, assign_fp(trial,a), mean(gamma), std(gamma), min(gamma), max(gamma), ...
            mean(xi), D_mean(trial,a), D_p10(trial,a), D_p50(trial,a), D_p90(trial,a), ...
            avg_qoe(trial,a), deadline_vio(trial,a), semantic_vio(trial,a));
    end
end

result.algorithms = algorithms;

result.sum_rate = mean(sum_rate, 1);
result.avg_qoe  = mean(avg_qoe, 1);

result.sum_rate_std = std(sum_rate, 0, 1);
result.avg_qoe_std  = std(avg_qoe, 0, 1);

result.deadline_vio     = mean(deadline_vio, 1);
result.semantic_vio     = mean(semantic_vio, 1);
result.deadline_vio_std = std(deadline_vio, 0, 1);
result.semantic_vio_std = std(semantic_vio, 0, 1);

result.D_mean = mean(D_mean, 1);
result.D_p10  = mean(D_p10, 1);
result.D_p50  = mean(D_p50, 1);
result.D_p90  = mean(D_p90, 1);

result.assign_fp = assign_fp;

result.seed = seed;
result.mc = mc;
result.p_dbw = p_dbw;
result.semantic_mode = semantic_mode;
result.table_path = table_path;
result.weights = weights;

run_id = save_results('run_once', result, p_dbw);
result.run_id = run_id;

fprintf('Run complete. Results saved with run_id: %s\n', run_id);
end

function fp = assignment_fingerprint(assign)
% Create a compact fingerprint for an assignment matrix/vector.
% Works for numeric arrays.

x = double(assign(:));
% remove NaN just in case
x = x(~isnan(x));
if isempty(x)
    fp = "empty";
    return;
end

% A few stable scalar features
s1 = sum(x);
s2 = sum(x.^2);
s3 = sum(abs(diff(x)));

% Add a simple rolling hash
h = uint64(1469598103934665603); % FNV offset basis
prime = uint64(1099511628211);
for k = 1:min(numel(x), 2000) % cap to keep it cheap
    v = uint64(mod(round(1e6*x(k)), 2^32));
    h = bitxor(h, v);
    h = h * prime;
end

fp = sprintf('%0.3f_%0.3f_%0.3f_%s', s1, s2, s3, dec2hex(h));
fp = string(fp);
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
            if isfield(info, 'skipped') && info.skipped
                assign = norm_based(cfg, ch);
            end
        case 'ga_placeholder'
            assign = ga_placeholder(cfg);
        otherwise
            assign = norm_based(cfg, ch);
    end
end
