function cfg = config(varargin)
%CONFIG Default simulation configuration.

cfg.num_cells = 1;
cfg.bs_positions = [0 0];

% K=12, L=4, k0=1
cfg.users_per_cell = 12;
cfg.ris_per_cell = 4;

cfg.nt = 4;       % 4 antennas at BS
cfg.n_ris = 36;   % 36 elements per RIS

cfg.k0 = 1;  % each RIS serves 1 user

% 噪声: -70 dBm
cfg.noise_dbm = -70;
cfg.noise_watts = 10.^((cfg.noise_dbm - 30) / 10);

cfg.bandwidth = 1e6;  % 1MHz

cfg.p_dbw_list = linspace(-25, -5, 8);

cfg.mc = 200;

cfg.m_k = 8;
cfg.rho = 1;

cfg.ris_phase_mode = 'align';

% RIS增益: 放大级联信道贡献 (必须保留！)
cfg.ris_gain = 1000;

cfg.random_hard_mask = false;

% 时延阈值 [严格, 宽松]
cfg.deadlines = [0.002 0.010];

% 语义失真阈值
cfg.dmax = 0.30;

% Sigmoid锐度
cfg.beta_d = 0.1;
cfg.beta_s = 0.1;

% 惩罚乘数
cfg.h_d = 1;
cfg.h_s = 1;

cfg.hard_ratio = 1.0;

% QoE权重
cfg.weights = [0.5 0.5; 0.5 0.5; 0.5 0.5];

% 语义映射模式: 'table' 使用DeepSC真实数据, 'proxy' 使用近似公式
cfg.semantic_mode = 'table';
cfg.semantic_table = 'semantic_tables/deepsc_table.csv';

% proxy模式参数 (备用)
cfg.proxy_a = 0.6;
cfg.proxy_b = 0.4;

% 路损指数
cfg.pathloss_exp = 3.2;
cfg.pathloss_exp_direct = 3.5;  % BS->UE
cfg.pathloss_exp_br = 2.2;      % BS->RIS
cfg.pathloss_exp_ru = 2.2;      % RIS->UE

cfg.eps = 1e-9;

% GA参数 - 时间公平配置
% Proposed改进版 ≈ 2-3ms, GA也限制在相似时间
cfg.ga_Np = 10;
cfg.ga_Niter = 2;

% 惩罚系数 (降低以使曲线更平滑)
cfg.b_d = 5;
cfg.b_s = 5;

% 应用覆盖参数
for idx = 1:2:numel(varargin)
    key = varargin{idx};
    cfg.(key) = varargin{idx + 1};
end

cfg.num_users = cfg.num_cells * cfg.users_per_cell;
cfg.num_ris = cfg.num_cells * cfg.ris_per_cell;
end
