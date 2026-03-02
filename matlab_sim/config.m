function cfg = config(varargin)
%CONFIG Default simulation configuration.

cfg.num_cells = 1;
cfg.bs_positions = [0 0];

% K=12, L=4, k0=1
cfg.users_per_cell = 12;
cfg.ris_per_cell = 4;
cfg.num_urgent = 4; % default urgent-user count for geometry/profile bootstrap

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
% 严格任务大约 3~4 ms，普通任务 10~12 ms
cfg.deadlines = [0.0035 0.011];

% 语义失真阈值: D <= dmax (即 xi >= 1-dmax)
cfg.dmax = 0.30;

% Sigmoid 锐度（越大越“陡”，在阈值附近更敏感）
cfg.beta_d = 2.0;
cfg.beta_s = 2.0;

% 超限线性惩罚系数（适中即可，避免成本过大）
% 为避免 QoE 成本数值过大，这里使用较小的线性惩罚，并对比率设置上限
cfg.h_d = 0.1;
cfg.h_s = 0.1;

% 比率上限：当 T_tx / d 或 D / dmax 远大于该值时，认为“极差”且惩罚饱和
cfg.max_ratio_d = 5;   % 时延最多视为 5 倍门限
cfg.max_ratio_s = 5;   % 语义失真最多视为 5 倍门限

cfg.hard_ratio = 1.0;

% QoE权重:
%   第1行: 紧迫/超实时任务 (更看重时延)
%   第2行: 实时交互任务 (时延与语义同等重要)
%   第3行: 检索/理解类任务 (更看重语义质量)
cfg.weights = [0.8 0.2;
               0.5 0.5;
               0.2 0.8];

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
% 提高种群规模与迭代次数，减小随机性、使 GA 收敛更稳定
cfg.ga_Np = 30;     % 原来为 10
cfg.ga_Niter = 8;   % 原来为 2

% RIS phase UQP+MM controls
cfg.ris_mm_iter = 6;          % MM inner iterations per RIS block
cfg.ris_mm_alpha = 1.0;       % initial damping factor for rollback search
cfg.ris_mm_alpha_min = 1/64;  % minimum damping before rejecting update
cfg.ris_mm_tol = 1e-9;        % monotonicity tolerance on proxy objective
cfg.ris_mm_private_scale = 1; % private-stream term scale in phase proxy objective
cfg.ris_mm_debug = false;     % print MM objective trace when true

% 惩罚系数 (减小以避免 QoE 数值过大)
cfg.b_d = 0.5;
cfg.b_s = 0.5;

% 应用覆盖参数
for idx = 1:2:numel(varargin)
    key = varargin{idx};
    cfg.(key) = varargin{idx + 1};
end

cfg.num_users = cfg.num_cells * cfg.users_per_cell;
cfg.num_ris = cfg.num_cells * cfg.ris_per_cell;
end
