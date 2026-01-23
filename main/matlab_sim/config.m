function cfg = config(varargin)
%CONFIG Default simulation configuration.

cfg.num_cells = 1;  % single-cell (multi-cell not modeled in channel.m)
cfg.bs_positions = [0 0];  % one BS at origin

cfg.users_per_cell = 16;
cfg.ris_per_cell = 3;

cfg.nt = 4;
cfg.n_ris = 36;

cfg.k0 = 1;

cfg.noise_dbm = -70;
cfg.noise_watts = 10.^((cfg.noise_dbm - 30) / 10);

cfg.bandwidth = 1e6;

cfg.p_dbw_list = [-10 -5 0];

cfg.mc = 100;

cfg.m_k = 8;
cfg.rho = 1;

cfg.ris_phase_mode = 'align';   % 'random' or 'align'

% ---------- NEW: RIS gain knob ----------
% This scales the cascaded (BS->RIS->UE) reflected channel contribution.
% If RIS effect is negligible, increase this (e.g., 500~2000).
cfg.ris_gain = 1000;

cfg.random_hard_mask = false;   % 见 qoe.m

cfg.deadlines = [0.002 0.008];

cfg.dmax = 0.08;
cfg.beta_d = 0.1;
cfg.beta_s = 0.1;

cfg.h_d = 1;
cfg.h_s = 1;

cfg.hard_ratio = 0.5;
cfg.b_d = 10;
cfg.b_s = 10;

cfg.weights = [0.8 0.2; 0.5 0.5; 0.2 0.8];

cfg.semantic_mode = 'proxy';
cfg.semantic_table = '';

cfg.proxy_a = 0.6;
cfg.proxy_b = 0.4;

cfg.pathloss_exp = 3.2;

cfg.eps = 1e-9;

for idx = 1:2:numel(varargin)
    key = varargin{idx};
    cfg.(key) = varargin{idx + 1};
end

% Sanity: bs_positions must match num_cells; truncate extras if any
if size(cfg.bs_positions, 1) < cfg.num_cells
    error('cfg.bs_positions must have at least cfg.num_cells rows.');
end
cfg.bs_positions = cfg.bs_positions(1:cfg.num_cells, :);
cfg.num_users = cfg.num_cells * cfg.users_per_cell;
cfg.num_ris = cfg.num_cells * cfg.ris_per_cell;
end