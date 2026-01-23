function run_id = save_results(tag, result, x_vals)
%SAVE_RESULTS Save MAT/CSV/JSON outputs.

if nargin < 3
    x_vals = [];
end

run_id = sprintf('%s_%s', tag, datestr(now, 'yyyymmdd_HHMMSS'));

if ~exist('results', 'dir')
    mkdir('results');
end

mat_path = fullfile('results', [run_id '_curves.mat']);
csv_path = fullfile('results', [run_id '_curves.csv']);
json_path = fullfile('results', [run_id '_metrics.json']);

output = struct();
output.run_id = run_id;
output.result = result;
output.x_vals = x_vals;

save(mat_path, '-struct', 'output');

if isfield(result, 'x_axis')
    headers = [{'x_val'}, strcat(result.algorithms, '_sum_rate'), strcat(result.algorithms, '_avg_qoe')];
    data = [x_vals(:), result.sum_rate, result.avg_qoe];
else
    headers = [{'x_val'}, strcat(result.algorithms, '_sum_rate'), strcat(result.algorithms, '_avg_qoe')];
    data = [x_vals(:), result.sum_rate(:).', result.avg_qoe(:).'];
end

writecell(headers, csv_path);

fid = fopen(csv_path, 'a');
for i = 1:size(data, 1)
    fprintf(fid, '%g', data(i, 1));
    for j = 2:size(data, 2)
        fprintf(fid, ',%g', data(i, j));
    end
    fprintf(fid, '\n');
end
fclose(fid);

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
