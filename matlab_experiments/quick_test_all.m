%QUICK_TEST_ALL 运行所有实验，生成全部图片
%
% 算法: random, norm, proposed, ga
% 输出图:
%   Sweep实验:
%     1. sweep_*_avg_qoe.png
%     2. sweep_*_sum_rate.png
%   Heterogeneous实验:
%     3. hetero_*_avg_qoe.png
%     4. hetero_*_urllc_qoe.png
%     5. hetero_*_semantic_qoe.png
%   Scarcity实验:
%     6. scarcity_*_avg_qoe.png
%   RIS Count实验:
%     7. ris_count_*_avg_qoe.png
%   GA Convergence:
%     8. ga_convergence_*.png
%   紧迫用户场景:
%     9. urgent_*_avg_qoe.png
%    10. urgent_*_urgent_qoe.png
%    11. urgent_*_normal_qoe.png
%    12. urgent_*_sum_rate.png
%    13. urgent_*_delay_violation.png
%    14. urgent_*_semantic_xi.png
%    15. urgent_*_sinr.png

clear; clc;

fprintf('========================================\n');
fprintf('运行所有实验 (生成全部图片)\n');
fprintf('========================================\n\n');

mc_default = 200;
base_seed = 42;

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
cd(proj_root);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');
addpath(fullfile(proj_root, 'matlab_experiments'), '-begin');

fprintf('MC = %d, Seed = %d\n', mc_default, base_seed);
fprintf('算法: random, norm, proposed, ga\n\n');

t_start = tic;

% 1. Sweep
fprintf('【1/7】运行 Sweep 实验...\n');
try
    sweep('mc', mc_default, 'seed', base_seed);
    fprintf('  ✓ sweep 完成 (2张图)\n');
catch ME
    fprintf('  ✗ sweep 失败: %s\n', ME.message);
end
fprintf('\n');

% 2. Heterogeneous
fprintf('【2/7】运行 Heterogeneous 实验...\n');
try
    sweep_advanced('heterogeneous', 'mc', mc_default, 'seed', base_seed);
    fprintf('  ✓ heterogeneous 完成 (3张图)\n');
catch ME
    fprintf('  ✗ heterogeneous 失败: %s\n', ME.message);
end
fprintf('\n');

% 3. Scarcity
fprintf('【3/7】运行 Scarcity 实验...\n');
try
    sweep_advanced('scarcity', 'mc', mc_default, 'seed', base_seed);
    fprintf('  ✓ scarcity 完成 (1张图)\n');
catch ME
    fprintf('  ✗ scarcity 失败: %s\n', ME.message);
end
fprintf('\n');

% 4. RIS Count
fprintf('【4/7】运行 RIS Count 实验...\n');
try
    sweep_advanced('ris_count', 'mc', mc_default, 'seed', base_seed);
    fprintf('  ✓ ris_count 完成 (1张图)\n');
catch ME
    fprintf('  ✗ ris_count 失败: %s\n', ME.message);
end
fprintf('\n');

% 5. GA Convergence
fprintf('【5/7】运行 GA Convergence 实验...\n');
try
    plot_ga_convergence('mc', 30, 'p_dbw', -15);
    fprintf('  ✓ GA convergence 完成 (1张图)\n');
catch ME
    fprintf('  ✗ GA convergence 失败: %s\n', ME.message);
end
fprintf('\n');

% 6. 紧迫用户场景
fprintf('【6/7】运行 紧迫用户场景 实验...\n');
try
    exp_urgent('mc', mc_default, 'seed', base_seed);
    fprintf('  ✓ urgent 完成 (7张图)\n');
catch ME
    fprintf('  ✗ urgent 失败: %s\n', ME.message);
end
fprintf('\n');

% 7. Constraint (不生成图，仅保存数据)
fprintf('【7/7】运行 Constraint 实验...\n');
try
    sweep_advanced('constraint', 'mc', mc_default, 'seed', base_seed);
    fprintf('  ✓ constraint 完成 (仅数据)\n');
catch ME
    fprintf('  ✗ constraint 失败: %s\n', ME.message);
end
fprintf('\n');

t_elapsed = toc(t_start);

fprintf('========================================\n');
fprintf('所有实验完成！\n');
fprintf('========================================\n');
fprintf('总耗时: %.1f 秒 (%.1f 分钟)\n', t_elapsed, t_elapsed/60);
fprintf('图片保存在: figures/ 目录\n\n');

fig_dir = fullfile(proj_root, 'figures');
if exist(fig_dir, 'dir')
    png_files = dir(fullfile(fig_dir, '*.png'));
    fprintf('目录中共有 %d 张图片\n', numel(png_files));
end

fprintf('\n预期生成图片清单:\n');
fprintf('  Sweep: avg_qoe, sum_rate (2张)\n');
fprintf('  Heterogeneous: avg_qoe, urllc_qoe, semantic_qoe (3张)\n');
fprintf('  Scarcity: avg_qoe (1张)\n');
fprintf('  RIS Count: avg_qoe (1张)\n');
fprintf('  GA Convergence: 1张\n');
fprintf('  紧迫场景: avg_qoe, urgent_qoe, normal_qoe, sum_rate, delay_violation, semantic_xi, sinr (7张)\n');
fprintf('  总计: 15张图片\n');
