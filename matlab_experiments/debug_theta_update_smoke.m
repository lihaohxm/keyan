function debug_theta_update_smoke()
%DEBUG_THETA_UPDATE_SMOKE Single-call smoke script for RIS MM phase update.
% This script is for manual debugging only (not auto-run).

this_file = mfilename('fullpath');
proj_root = fileparts(fileparts(this_file));
addpath(fullfile(proj_root, 'matlab_sim'), '-begin');
addpath(fullfile(proj_root, 'matlab_sim', 'matching'), '-begin');

cfg = config();
cfg.num_users = 6;
cfg.users_per_cell = 6;
cfg.num_ris = 2;
cfg.ris_per_cell = 2;
cfg.k0 = 2;
cfg.ris_mm_iter = 8;
cfg.ris_mm_debug = true;

seed = 123;
geom = geometry(cfg, seed);
ch = channel(cfg, geom, seed);
profile = build_profile_urgent_normal(cfg, geom, struct());

assign = zeros(cfg.num_users, 1);
assign(1:2) = 1;
assign(3:4) = 2;

p_dbw = -12;
h_eff = effective_channel(cfg, ch, assign, ch.theta);
[V, ~, ~, ~, ~, ~] = rsma_wmmse(cfg, h_eff, p_dbw, ones(cfg.num_users, 1), 4);
weights = ones(cfg.num_users, 1);

[theta_all, info] = ris_phase_mm(cfg, ch, assign, V, weights, cfg.ris_mm_iter, ch.theta);

fprintf('\n[smoke] theta size = %dx%d\n', size(theta_all,1), size(theta_all,2));
for l = 1:cfg.num_ris
    hist_l = info.obj_history_by_ris{l};
    if isempty(hist_l)
        fprintf('[smoke] RIS-%d: no associated users\n', l);
    else
        fprintf('[smoke] RIS-%d proxy obj seq: %s\n', l, mat2str(hist_l.', 6));
    end
end

% Optional end-to-end check with updated theta
sol = struct('assign', assign, 'theta_all', theta_all, 'V', V);
out = evaluate_system_rsma(cfg, ch, geom, sol, profile, struct('debug_eval', true));
fprintf('[smoke] final avg_qoe=%.6f, sum_rate=%.6f Mbps\n', out.avg_qoe, out.sum_rate_bps/1e6);
end
