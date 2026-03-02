function run_sanity_check()
%RUN_SANITY_CHECK 快速验证场景设置和 UA-QoE-AO 是否正常工作
% 在 MATLAB 命令行运行: run_sanity_check

    root = fileparts(mfilename('fullpath'));

    % 添加所有依赖路径
    addpath(fullfile(root, 'matlab_sim'));
    addpath(fullfile(root, 'matlab_sim', 'matching'));
    addpath(fullfile(root, 'matlab_experiments'));
    addpath(fullfile(root, 'matlab_scripts'));

    fprintf('=== 开始 Sanity Check (MC=2, p=-15 dBW) ===\n');

    % 运行单次实验
    result = run_once('seed',1,'mc',2,'p_dbw',-15,'semantic_mode','table');

    fprintf('\n=== 运行成功 ===\n');
    fprintf('算法: %s\n', strjoin(result.algorithms, ', '));
    fprintf('Avg QoE: %s\n', mat2str(result.avg_qoe, 6));
    fprintf('Sum-rate: %s\n', mat2str(result.sum_rate, 6));
end
