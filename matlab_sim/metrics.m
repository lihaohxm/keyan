function stats = metrics(values)
%METRICS Compute mean/std for vector or matrix.

stats.mean = mean(values, 1);
stats.std = std(values, 0, 1);
end
