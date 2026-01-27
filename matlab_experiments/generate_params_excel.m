%GENERATE_PARAMS_EXCEL 生成实验参数设置Excel文件
%
% 输出: results/experiment_parameters.xlsx

clear; clc;

this_file = mfilename('fullpath');
this_dir = fileparts(this_file);
proj_root = fileparts(this_dir);
cd(proj_root);

addpath(fullfile(proj_root, 'matlab_sim'), '-begin');

% 获取默认配置
cfg = config();

% 创建输出目录
out_dir = fullfile(proj_root, 'results');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
out_file = fullfile(out_dir, 'experiment_parameters.xlsx');

% 删除已存在的文件
if exist(out_file, 'file')
    delete(out_file);
end

fprintf('生成实验参数Excel文件...\n');

%% Sheet 1: 公共参数
common_params = {
    '参数', '符号', '值', '单位', '说明';
    'BS天线数', 'N_t', cfg.nt, '', '基站天线数量';
    'RIS元素数', 'N', cfg.n_ris, '', '每个RIS的反射元素数';
    'k0', 'k_0', cfg.k0, '', '每个RIS服务的用户数';
    '带宽', 'B', cfg.bandwidth/1e6, 'MHz', '系统带宽';
    '噪声功率', 'σ²', cfg.noise_dbm, 'dBm', '接收机噪声功率';
    '语义包大小', 'M', cfg.m_k, 'symbols', 'DeepSC编码符号数';
    '压缩比', 'ρ', cfg.rho, '', '语义压缩比';
    '时延阈值(严格)', 'T_max^strict', cfg.deadlines(1)*1000, 'ms', 'URLLC用户时延约束';
    '时延阈值(宽松)', 'T_max^loose', cfg.deadlines(2)*1000, 'ms', '普通用户时延约束';
    '语义失真阈值', 'D_max', cfg.dmax, '', '语义失真上限';
    '语义相似度阈值', 'ξ_th', 1-cfg.dmax, '', '语义相似度下限(1-D_max)';
    '路损指数(BS→UE)', 'α_d', cfg.pathloss_exp_direct, '', '直接链路路损指数';
    '路损指数(BS→RIS)', 'α_br', cfg.pathloss_exp_br, '', 'BS到RIS链路路损指数';
    '路损指数(RIS→UE)', 'α_ru', cfg.pathloss_exp_ru, '', 'RIS到UE链路路损指数';
    'RIS增益', 'G_RIS', cfg.ris_gain, '', 'RIS级联信道增益';
    'GA种群大小', 'N_p', cfg.ga_Np, '', '遗传算法种群';
    'GA迭代次数', 'N_iter', cfg.ga_Niter, '', '遗传算法迭代';
    'MC次数(默认)', 'MC', cfg.mc, '', '蒙特卡洛仿真次数';
};

writecell(common_params, out_file, 'Sheet', '公共参数');
fprintf('  ✓ 公共参数\n');

%% Sheet 2: 各实验参数对比
exp_compare = {
    '实验', 'Sweep', 'Heterogeneous', 'Scarcity', 'RIS Count', 'GA Convergence', '紧迫场景';
    '用户数 K', 12, 6, '3-8 (扫描)', 6, 12, 12;
    'RIS数 L', 4, 4, 4, '2-10 (扫描)', 4, 4;
    'k0', 1, 1, 1, 1, 1, 1;
    'RIS元素 N', 36, 36, 36, 36, 36, 36;
    '功率范围 (dBW)', '[-25,-5]', '[-25,-5]', '-15 (固定)', '-15 (固定)', '-15 (固定)', '[-25,-5]';
    '功率采样点', 8, 8, 1, 1, 1, 8;
    'MC次数', 200, 200, 200, 200, 30, 200;
    'QoE权重 w_d', 0.5, '异构', 0.5, 0.5, 0.5, 0.5;
    'QoE权重 w_s', 0.5, '异构', 0.5, 0.5, 0.5, 0.5;
    '输出图数', 2, 3, 1, 1, 1, 7;
};

writecell(exp_compare, out_file, 'Sheet', '实验参数对比');
fprintf('  ✓ 实验参数对比\n');

%% Sheet 3: Heterogeneous用户权重
hetero_weights = {
    '用户编号', '用户类型', 'w_d (时延权重)', 'w_s (语义权重)', '说明';
    '1-2', 'URLLC', 0.8, 0.2, '时延敏感型用户';
    '3-4', 'Semantic', 0.2, 0.8, '语义敏感型用户';
    '5-6', 'Best Effort', 0.5, 0.5, '均衡型用户';
};

writecell(hetero_weights, out_file, 'Sheet', 'Heterogeneous用户');
fprintf('  ✓ Heterogeneous用户权重\n');

%% Sheet 4: 紧迫场景参数
urgent_params = {
    '参数', '值', '说明';
    '=== 基础配置 ===', '', '';
    '总用户数 K', 12, '';
    'RIS数量 L', 4, '';
    '每RIS容量 k0', 1, '';
    'RIS元素数 N', 36, '';
    '功率范围', '[-25, -5] dBW', '8个采样点';
    'MC次数', 200, '';
    'QoE权重', 'w_d=0.5, w_s=0.5', '';
    '语义门槛 xi_th', 0.70, 'dmax=0.30';
    '', '', '';
    '=== 紧迫用户 (1-4) ===', '', '';
    '用户数量', 4, '边缘用户';
    '距离范围', '120-180 m', '远离BS';
    '方位角分布', '集中在一个扇区', 'urgent_sector ± 0.15';
    '路损指数 α', 4.0, 'NLoS (遮挡严重)';
    '信道条件', 'NLoS', '直连信道极差';
    '', '', '';
    '=== 普通用户 (5-12) ===', '', '';
    '用户数量', 8, '近距用户';
    '距离范围', '30-70 m', '靠近BS';
    '方位角分布', '均匀分布', '0-360°';
    '路损指数 α', 2.8, 'LoS (直视)';
    '信道条件', 'LoS', '直连信道正常';
    '', '', '';
    '=== RIS部署 ===', '', '';
    'RIS距离', '80-100 m', '部署在紧迫用户方向';
    'RIS方位角', '紧迫用户扇区', '扇形分布覆盖边缘';
    '路损指数 (BS→RIS)', 2.2, 'LoS';
    '路损指数 (RIS→UE)', 2.5, '';
};

writecell(urgent_params, out_file, 'Sheet', '紧迫场景');
fprintf('  ✓ 紧迫场景参数\n');

%% Sheet 5: 输出图片清单
output_figures = {
    '序号', '实验', '图片文件名', 'X轴', 'Y轴', '说明';
    1, 'Sweep', 'sweep_*_avg_qoe.png', 'p (dBW)', 'Avg QoE Cost', '平均QoE代价随功率变化';
    2, 'Sweep', 'sweep_*_sum_rate.png', 'p (dBW)', 'Sum-rate', '系统和速率随功率变化';
    3, 'Heterogeneous', 'hetero_*_avg_qoe.png', 'p (dBW)', 'Avg QoE Cost', '异构用户平均QoE';
    4, 'Heterogeneous', 'hetero_*_urllc_qoe.png', 'p (dBW)', 'URLLC QoE Cost', 'URLLC用户QoE';
    5, 'Heterogeneous', 'hetero_*_semantic_qoe.png', 'p (dBW)', 'Semantic QoE Cost', '语义用户QoE';
    6, 'Scarcity', 'scarcity_*_avg_qoe.png', 'K (用户数)', 'Avg QoE Cost', '资源稀缺性影响';
    7, 'RIS Count', 'ris_count_*_avg_qoe.png', 'L (RIS数)', 'Avg QoE Cost', 'RIS数量影响';
    8, 'GA Convergence', 'ga_convergence_*.png', 'Time/Iteration', 'QoE Cost', 'GA收敛曲线';
    9, '紧迫场景', 'urgent_*_avg_qoe.png', 'p (dBW)', 'Avg QoE Cost', '全体用户平均QoE';
    10, '紧迫场景', 'urgent_*_urgent_qoe.png', 'p (dBW)', 'Urgent QoE Cost', '紧迫用户QoE';
    11, '紧迫场景', 'urgent_*_normal_qoe.png', 'p (dBW)', 'Normal QoE Cost', '普通用户QoE';
    12, '紧迫场景', 'urgent_*_sum_rate.png', 'p (dBW)', 'Sum-rate', '系统和速率';
    13, '紧迫场景', 'urgent_*_delay_violation.png', 'p (dBW)', 'Violation Rate (%)', '时延违约率';
    14, '紧迫场景', 'urgent_*_semantic_xi.png', 'p (dBW)', 'ξ (Semantic Similarity)', '语义相似度分布';
    15, '紧迫场景', 'urgent_*_sinr.png', 'p (dBW)', 'SINR (dB)', 'SINR分布';
};

writecell(output_figures, out_file, 'Sheet', '输出图片清单');
fprintf('  ✓ 输出图片清单\n');

%% Sheet 6: 算法列表
algorithms = {
    '算法标识', '算法名称', '类型', '说明';
    'random', 'Random Matching', '基线', '随机分配用户到RIS';
    'norm', 'Norm-based Matching', '基线', '基于信道范数的贪婪匹配';
    'proposed', 'QoE-aware Greedy', '提出算法', 'QoE感知的贪婪匹配算法';
    'ga', 'Genetic Algorithm', '对比算法', '遗传算法全局优化';
};

writecell(algorithms, out_file, 'Sheet', '算法列表');
fprintf('  ✓ 算法列表\n');

fprintf('\n========================================\n');
fprintf('Excel文件已生成: %s\n', out_file);
fprintf('========================================\n');
fprintf('包含工作表:\n');
fprintf('  1. 公共参数\n');
fprintf('  2. 实验参数对比\n');
fprintf('  3. Heterogeneous用户\n');
fprintf('  4. 紧迫场景\n');
fprintf('  5. 输出图片清单\n');
fprintf('  6. 算法列表\n');
