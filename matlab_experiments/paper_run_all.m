function run_id = paper_run_all(varargin)
%PAPER_RUN_ALL One-click runner for all paper figures and results.
% Usage:
% paper_run_all('mc_sweep',50,'mc_cdf',200,'mc_ablation',200, ...
%               'seed',42,'p_dbw_cdf',-20,'p_dbw_ablation',-20)

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
addpath(fullfile(proj_root, 'matlab_experiments'), '-begin');

p = inputParser;
addParameter(p, 'mc_sweep', 50);
addParameter(p, 'mc_cdf', 200);
addParameter(p, 'mc_ablation', 200);
addParameter(p, 'seed', 42);
addParameter(p, 'p_dbw_cdf', -20);
addParameter(p, 'p_dbw_ablation', -20);
addParameter(p, 'run_ris_count', true);
addParameter(p, 'ris_list', [8 16 32 48 64 80 96 112 128]);
addParameter(p, 'p_dbw_ris', -8);
addParameter(p, 'mc_ris', []);
parse(p, varargin{:});

mc_sweep = p.Results.mc_sweep;
mc_cdf = p.Results.mc_cdf;
mc_ablation = p.Results.mc_ablation;
seed = p.Results.seed;
p_dbw_cdf = p.Results.p_dbw_cdf;
p_dbw_ablation = p.Results.p_dbw_ablation;
run_ris_count = p.Results.run_ris_count;
ris_list = p.Results.ris_list;
p_dbw_ris = p.Results.p_dbw_ris;
mc_ris = p.Results.mc_ris;
if isempty(mc_ris)
    mc_ris = mc_sweep;
end

fprintf('========================================\n');
fprintf('paper_run_all start\n');
fprintf('mc_sweep=%d, mc_cdf=%d, mc_ablation=%d\n', mc_sweep, mc_cdf, mc_ablation);
fprintf('seed=%d, p_dbw_cdf=%.2f, p_dbw_ablation=%.2f\n', seed, p_dbw_cdf, p_dbw_ablation);
fprintf('run_ris_count=%d, mc_ris=%d, p_dbw_ris=%.2f, ris_list=%s\n', ...
    logical(run_ris_count), mc_ris, p_dbw_ris, mat2str(ris_list));
fprintf('========================================\n');

fprintf('\n[1/3] Running paper_sweep_power...\n');
paper_sweep_power('mc', mc_sweep, 'seed', seed);
ensure_result_json_metrics(proj_root, 'paper_sweep_power_fixed.json');

% fprintf('\n[2/3] Running paper_cdf...\n');
% paper_cdf('mc', mc_cdf, 'p_dbw', p_dbw_cdf, 'seed', seed);

fprintf('\n[3/3] Running paper_ablation...\n');
paper_ablation('mc', mc_ablation, 'p_dbw', p_dbw_ablation, 'seed', seed);

if run_ris_count
    fprintf('\n[4/4] Running paper_sweep_ris_count...\n');
    paper_sweep_ris_count('mc', mc_ris, 'seed', seed, 'p_dbw', p_dbw_ris, 'ris_list', ris_list);
    ensure_result_json_metrics(proj_root, 'ris_count_fixed.json');
end

run_id = datestr(now, 'yyyymmdd_HHMMSS');
fprintf('\nAll done. run_id=%s\n', run_id);
end

function ensure_result_json_metrics(proj_root, file_name)
json_path = fullfile(proj_root, 'results', file_name);
if ~exist(json_path, 'file')
    return;
end
obj = jsondecode(fileread(json_path));
if ~isfield(obj, 'metrics') || ~isstruct(obj.metrics)
    return;
end
obj.metrics = ensure_urgent_rate_metrics(obj.metrics);
write_text_file(json_path, jsonencode(obj));
end

function stats = ensure_urgent_rate_metrics(stats)
if ~isfield(stats, 'sum_rate')
    return;
end
if ~isfield(stats.sum_rate, 'mean') || ~isfield(stats.sum_rate, 'ci95')
    return;
end
mu = stats.sum_rate.mean;
ci = stats.sum_rate.ci95;
if ~isfield(stats, 'urgent_sum_rate')
    % Unit: bps. Default NaN to keep backward compatibility on legacy files.
    stats.urgent_sum_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
if ~isfield(stats, 'urgent_avg_rate')
    % Unit: bps. Default NaN to keep backward compatibility on legacy files.
    stats.urgent_avg_rate = struct('mean', nan(size(mu)), 'ci95', nan(size(ci)));
end
end

function write_text_file(path_name, txt)
fid = fopen(path_name, 'w');
if fid < 0
    error('Cannot open file for writing: %s', path_name);
end
cleanup_obj = onCleanup(@() fclose(fid));
fprintf(fid, '%s', txt);
clear cleanup_obj;
end
