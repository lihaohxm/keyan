function cfg = config(varargin)
%CONFIG Default simulation configuration.

cfg.num_cells = 1;
cfg.bs_positions = [0 0];

% K=12, L=4, k0=1
cfg.users_per_cell = 12;
cfg.ris_per_cell = 4;
cfg.num_urgent = 4; % number of high-priority users selected per trial

cfg.nt = 4;       % 4 antennas at BS
cfg.n_ris = 36;   % 36 elements per RIS

cfg.k0 = 2;  % users served per RIS

% Noise: -70 dBm
cfg.noise_dbm = -70;
cfg.noise_watts = 10.^((cfg.noise_dbm - 30) / 10);

cfg.bandwidth = 1e6;  % 1 MHz

cfg.p_dbw_list = linspace(-25, -5, 8);

cfg.mc = 200;

cfg.m_k = 16;
cfg.rho = 30;

% Task-profile generation
cfg.task_type_probs = [0.25 0.45 0.30];           % [latency, dual, semantic]
cfg.task_deadlines = [0.006 0.010 0.016];         % seconds
cfg.task_dmax = [0.72 0.62 0.55];                 % strict-but-reachable distortion bounds
cfg.task_weight_rows = [1 2 3];
cfg.task_M = [4 8 16];                            % smaller packets for latency tasks, richer packets for semantic tasks
cfg.priority_weights = [0.45 0.40 0.15];         % [delay, semantic, channel-risk]
cfg.priority_preference_gain = 0.75;
cfg.priority_channel_softness = 0.50;
cfg.task_names = {'latency_critical', 'dual_critical', 'semantic_critical'};
cfg.task_seed_offset = 7919;

% Geometry generation
cfg.user_radius_min = 30;
cfg.user_radius_max = 180;
cfg.ris_radius_min = 70;
cfg.ris_radius_max = 110;

cfg.ris_phase_mode = 'align';
cfg.ris_gain = 30;

cfg.random_hard_mask = false;

% Delay thresholds [urgent, normal]
cfg.deadlines = [0.006 0.016];

% Semantic distortion threshold D <= dmax (equiv. xi >= 1-dmax)
cfg.dmax = 0.62;

cfg.beta_d = 2.0;
cfg.beta_s = 2.0;

% QoE exceedance penalties
cfg.h_d = 0.1;
cfg.h_s = 1.0;
cfg.semantic_penalty_power = 3.0;
cfg.delay_ratio_center = 0.75;
cfg.semantic_ratio_center = 0.78;
cfg.delay_soft_ratio_trigger = 0.80;
cfg.delay_soft_penalty = 0.03;
cfg.semantic_soft_ratio_trigger = 0.75;
cfg.semantic_soft_penalty = 0.20;

% Ratio clipping to avoid extreme costs
cfg.max_ratio_d = 5;
cfg.max_ratio_s = 5;
cfg.hard_ratio = 1.0;

% Task preference rows:
% row 1: urgent / latency-sensitive
% row 2: balanced interactive
% row 3: semantic-heavy understanding / text-like
cfg.weights = [0.55 0.45;
               0.45 0.55;
               0.25 0.75];

% Task-aware WMMSE weighting
cfg.qweight_urgent_bonus = 0.05;
cfg.qweight_priority_score_gain = 0.25;
cfg.qweight_task_pressure_gain = 0.55;
cfg.qweight_delay_tightness_gain = 0.15;
cfg.qweight_semantic_tightness_gain = 0.60;
cfg.qweight_delay_excess_gain = 0.20;
cfg.qweight_semantic_excess_gain = 1.00;
cfg.qweight_gradient_gain = 0.15;
cfg.qweight_excess_cap = 1.5;
cfg.qweight_tightness_cap = 1.5;
cfg.qweight_min = 0.7;
cfg.qweight_max = 2.6;
cfg.qweight_bias_floor = 0.35;
cfg.qweight_stress_trigger = 0.85;

% Endogenous common/private structure adaptation
cfg.common_power_cap_base = 0.38;
cfg.common_power_cap_min = 0.30;
cfg.common_power_cap_upper = 0.50;
cfg.common_cap_gain_weight = 0.20;
cfg.common_cap_semantic_weight = 0.03;
cfg.common_cap_delay_weight = 0.02;
cfg.common_rate_floor_urgent = 0.003e6;
cfg.common_rate_floor_share = 0.04;
cfg.common_rate_urgent_bonus = 0.10;
cfg.common_excess_penalty_weight = 1.0;

% Theta-step defaults: keep the AO loop driven by the same endogenous cost
% instead of legacy guarded polish heuristics.
cfg.theta_pre_refit_guard_urgent_qoe_drop = inf;
cfg.theta_polish_rounds = 0;

cfg.semantic_mode = 'table';
cfg.semantic_table = 'semantic_tables/deepsc_table.csv';

% Proxy semantic parameters (fallback mode)
cfg.proxy_a = 0.6;
cfg.proxy_b = 0.4;

% Pathloss exponents
cfg.pathloss_exp = 3.2;
cfg.pathloss_exp_direct = 3.5;  % BS->UE
cfg.pathloss_exp_br = 2.2;      % BS->RIS
cfg.pathloss_exp_ru = 2.2;      % RIS->UE

cfg.eps = 1e-9;

% GA settings
cfg.ga_Np = 30;
cfg.ga_Niter = 8;

% RIS phase UQP+MM controls
cfg.ris_mm_iter = 12;
cfg.ris_mm_alpha_min = 1/64;
cfg.ris_mm_tol = 1e-9;
cfg.ris_mm_private_scale = 1;
cfg.ris_mm_debug = false;

% Sigmoid sharpness
cfg.b_d = 10.0;
cfg.b_s = 10.0;

% AO parameters
cfg.ao_accept_rel = 5e-4;
cfg.ao_max_outer_iter = 15;

% Apply overrides
for idx = 1:2:numel(varargin)
    key = varargin{idx};
    cfg.(key) = varargin{idx + 1};
end

cfg.num_users = cfg.num_cells * cfg.users_per_cell;
cfg.num_ris = cfg.num_cells * cfg.ris_per_cell;
end
