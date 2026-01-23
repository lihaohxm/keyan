function cfg = config(varargin)
%CONFIG Default simulation configuration.

cfg.num_cells = 2;
cfg.bs_positions = [0 0; 0 320];

cfg.users_per_cell = 8;
cfg.ris_per_cell = 6;

cfg.nt = 4;
cfg.n_ris = 36;

cfg.k0 = 4;

cfg.noise_dbm = -70;
cfg.noise_watts = 10.^((cfg.noise_dbm - 30) / 10);

cfg.bandwidth = 1e6;

cfg.p_dbw_list = [-10 -5 0];

cfg.mc = 100;

cfg.m_k = 8;
cfg.rho = 1;

cfg.deadlines = [0.01 0.03];

cfg.dmax = 0.2;
cfg.beta_d = 1;
cfg.beta_s = 1;

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

cfg.num_users = cfg.num_cells * cfg.users_per_cell;
cfg.num_ris = cfg.num_cells * cfg.ris_per_cell;
end
