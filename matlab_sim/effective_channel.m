function h_eff = effective_channel(cfg, ch, assign, theta_all)
%EFFECTIVE_CHANNEL Compute effective channel with assignment.
%
% cfg.ris_phase_mode:
%   'random' : use pre-generated ch.theta(:,l)
%   'align'  : per-user phase alignment baseline
%
% cfg.ris_gain: scales RIS reflected contribution

num_users = cfg.num_users;

h_eff = ch.h_d;

if ~isfield(cfg, 'ris_phase_mode') || isempty(cfg.ris_phase_mode)
    cfg.ris_phase_mode = 'random';
end
if ~isfield(cfg, 'ris_gain') || isempty(cfg.ris_gain)
    cfg.ris_gain = 1000;
end

use_external_theta = (nargin >= 4) && ~isempty(theta_all);
if use_external_theta
    if size(theta_all, 2) ~= cfg.num_ris
        error('theta_all must be N x L with L = cfg.num_ris.');
    end
end

for k = 1:num_users
    l = assign(k);

    if l > 0 && l <= cfg.num_ris
        h_ris = ch.H_ris(:, k, l);
        G_l = ch.G(:, :, l);

        if use_external_theta
            theta = theta_all(:, l);
        else
            if strcmpi(cfg.ris_phase_mode, 'random')
                theta = ch.theta(:, l);
            elseif strcmpi(cfg.ris_phase_mode, 'align')
                hdk = ch.h_d(:, k);
                denom = norm(hdk) + cfg.eps;
                w0 = hdk / denom;
                proj = (w0' * G_l).' .* h_ris;
                theta = exp(-1j * angle(proj + cfg.eps));
            else
                theta = ch.theta(:, l);
            end
        end

        h_eff(:, k) = h_eff(:, k) + cfg.ris_gain * (G_l * (theta .* h_ris));
    end
end
end
