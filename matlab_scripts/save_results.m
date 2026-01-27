function run_id = save_results(tag, result, x_vals)
%SAVE_RESULTS Save MAT/CSV/JSON outputs.
% - Always saves: results/<run_id>_curves.mat/.csv and <run_id>_metrics.json
% - Supports:
%   (A) Multi-algorithms mode (preferred): result.algorithms exists and
%       size(result.sum_rate,1)==numel(x_vals), size(result.sum_rate,2)==numel(algorithms)
%       -> outputs per-algorithm columns: *_sum_rate, *_avg_qoe, optional *_deadline_vio, *_semantic_vio
%   (B) Fallback point mode: single sum_rate/avg_qoe vectors

if nargin < 3
    x_vals = [];
end

run_id = sprintf('%s_%s', tag, datestr(now, 'yyyymmdd_HHMMSS'));

if ~exist('results', 'dir')
    mkdir('results');
end

mat_path  = fullfile('results', [run_id '_curves.mat']);
csv_path  = fullfile('results', [run_id '_curves.csv']);
json_path = fullfile('results', [run_id '_metrics.json']);

% ---- Save MAT (raw) ----
output = struct();
output.run_id = run_id;
output.result = result;
output.x_vals = x_vals;
save(mat_path, '-struct', 'output');

% ---- Decide mode ----
has_algs = isfield(result, 'algorithms') && ~isempty(result.algorithms);
has_x    = ~isempty(x_vals);

sr = [];
aq = [];
if isfield(result, 'sum_rate'), sr = result.sum_rate; end
if isfield(result, 'avg_qoe'),  aq = result.avg_qoe;  end

% Optional metrics
has_dv = isfield(result, 'deadline_vio');
has_sv = isfield(result, 'semantic_vio');
dv = [];
sv = [];
if has_dv, dv = result.deadline_vio; end
if has_sv, sv = result.semantic_vio; end

% Ensure numeric when needed
if iscell(sr), sr = cell2mat(sr); end
if iscell(aq), aq = cell2mat(aq); end
if has_dv && iscell(dv), dv = cell2mat(dv); end
if has_sv && iscell(sv), sv = cell2mat(sv); end

use_multi_alg = false;
if has_algs && has_x && isnumeric(sr) && isnumeric(aq)
    A = numel(result.algorithms);
    if ismatrix(sr) && ismatrix(aq) && size(sr,1) == numel(x_vals) && size(sr,2) == A && ...
                       size(aq,1) == numel(x_vals) && size(aq,2) == A
        use_multi_alg = true;
    end
end

% ---- Build CSV headers + data ----
if use_multi_alg
    algs = result.algorithms(:).';
    headers = [{'x_val'}, strcat(algs, '_sum_rate'), strcat(algs, '_avg_qoe')];

    x_col = x_vals(:);

    % Pad rows to match x_col
    n = max([size(sr,1), size(aq,1), numel(x_col)]);
    x_col(end+1:n,1) = NaN;
    if size(sr,1) < n, sr(end+1:n, :) = NaN; end
    if size(aq,1) < n, aq(end+1:n, :) = NaN; end

    data = [x_col, sr, aq];

    % Optional add: deadline/semantic violation
    if has_dv
        if size(dv,1) < n, dv(end+1:n, :) = NaN; end
        headers = [headers, strcat(algs, '_deadline_vio')];
        data = [data, dv];
    end
    if has_sv
        if size(sv,1) < n, sv(end+1:n, :) = NaN; end
        headers = [headers, strcat(algs, '_semantic_vio')];
        data = [data, sv];
    end

else
    % ===== Fallback point mode =====
    headers = {'x_val','sum_rate','avg_qoe'};
    x_col = x_vals(:);

    if isempty(sr) || isempty(aq)
        error('save_results: result.sum_rate and result.avg_qoe are required.');
    end

    % Flatten
    sr_col = sr(:);
    aq_col = aq(:);

    n = max([numel(x_col), numel(sr_col), numel(aq_col)]);
    x_col(end+1:n,1)  = NaN;
    sr_col(end+1:n,1) = NaN;
    aq_col(end+1:n,1) = NaN;

    data = [x_col, sr_col, aq_col];

    if has_dv
        dv_col = dv(:);
        dv_col(end+1:n,1) = NaN;
        headers = [headers, {'deadline_vio'}];
        data = [data, dv_col];
    end
    if has_sv
        sv_col = sv(:);
        sv_col(end+1:n,1) = NaN;
        headers = [headers, {'semantic_vio'}];
        data = [data, sv_col];
    end
end

% ---- Write CSV ----
writecell(headers, csv_path);

fid = fopen(csv_path, 'a');
for i = 1:size(data, 1)
    % first column
    fprintf(fid, '%g', data(i, 1));
    for j = 2:size(data, 2)
        fprintf(fid, ',%g', data(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);

% ---- Write JSON (meta) ----
meta = struct();
meta.run_id = run_id;
meta.tag = tag;
meta.timestamp = datestr(now, 31);
meta.result = result;
meta.x_vals = x_vals;

fid = fopen(json_path, 'w');
fprintf(fid, '%s', jsonencode(meta));
fclose(fid);

end
