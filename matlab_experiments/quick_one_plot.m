%QUICK_ONE_PLOT 快速生成一张图，用于查看绘图风格
clear; clc;

% Setup paths
this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
cd(proj_root);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');
addpath(fullfile(proj_root, 'matlab_scripts'), '-begin');

fprintf('快速生成一张图...\n');

% 只运行 sweep 实验，mc=10 快速出图
sweep('mc', 10, 'seed', 42);

fprintf('\n图片已保存到 figures/ 目录\n');
