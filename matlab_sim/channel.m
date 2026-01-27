function ch = channel(cfg, geom, seed)
%CHANNEL Generate large/small-scale channels.
%
% 简化路损模型: PL = d^(-alpha)
% 这是标准的归一化路损模型，物理合理且易于调试。

if nargin < 3
    seed = 1;
end

rng(seed, 'twister');

num_users = cfg.num_users;
num_ris = cfg.num_ris;

h_d = zeros(cfg.nt, num_users);
G = zeros(cfg.nt, cfg.n_ris, num_ris);
H_ris = zeros(cfg.n_ris, num_users, num_ris);

% 路损指数
alpha_direct = get_field(cfg, 'pathloss_exp_direct', 3.5);  % BS->UE
alpha_br = get_field(cfg, 'pathloss_exp_br', 2.2);          % BS->RIS (LoS)
alpha_ru = get_field(cfg, 'pathloss_exp_ru', 2.2);          % RIS->UE (LoS)

% BS位置
bs_pos = geom.bs;
if numel(bs_pos) == 3, bs_pos = bs_pos(1:2); end

% 直连信道 BS->UE
for k = 1:num_users
    ue_pos = geom.ue(k, :);
    if numel(ue_pos) == 3, ue_pos = ue_pos(1:2); end
    d = norm(ue_pos - bs_pos) + cfg.eps;
    
    % 简化路损: PL = d^(-alpha)
    pathloss = d.^(-alpha_direct);
    h_d(:, k) = sqrt(pathloss/2) * (randn(cfg.nt, 1) + 1j * randn(cfg.nt, 1));
end

% 级联信道 BS->RIS->UE
for l = 1:num_ris
    ris_pos = geom.ris(l, :);
    if numel(ris_pos) == 3, ris_pos = ris_pos(1:2); end
    
    d_br = norm(ris_pos - bs_pos) + cfg.eps;
    pl_br = d_br.^(-alpha_br);
    G(:, :, l) = sqrt(pl_br/2) * (randn(cfg.nt, cfg.n_ris) + 1j * randn(cfg.nt, cfg.n_ris));

    for k = 1:num_users
        ue_pos = geom.ue(k, :);
        if numel(ue_pos) == 3, ue_pos = ue_pos(1:2); end
        d_ru = norm(ue_pos - ris_pos) + cfg.eps;
        pl_ru = d_ru.^(-alpha_ru);
        H_ris(:, k, l) = sqrt(pl_ru/2) * (randn(cfg.n_ris, 1) + 1j * randn(cfg.n_ris, 1));
    end
end

theta = exp(1j * 2 * pi * rand(cfg.n_ris, num_ris));

ch.h_d = h_d;
ch.G = G;
ch.H_ris = H_ris;
ch.theta = theta;
end

function v = get_field(s, f, d)
    if isfield(s, f), v = s.(f); else, v = d; end
end
