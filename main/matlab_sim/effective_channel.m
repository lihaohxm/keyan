function h_eff = effective_channel(cfg, ch, assign)
%EFFECTIVE_CHANNEL Compute effective channel with assignment.
% cfg.ris_phase_mode:
%   'random' (default) : use pre-generated ch.theta(:,l)
%   'align'            : per-user phase alignment baseline
%
% cfg.ris_gain:
%   scales RIS reflected contribution to avoid negligible cascaded pathloss.

num_users = cfg.num_users;

h_eff = ch.h_d;

% defaults
if ~isfield(cfg, 'ris_phase_mode') || isempty(cfg.ris_phase_mode)
    cfg.ris_phase_mode = 'random';
end
if ~isfield(cfg, 'ris_gain') || isempty(cfg.ris_gain)
    cfg.ris_gain = 1;
end

for k = 1:num_users
    l = assign(k);

    % l<=0 means "no RIS used for this user"
    if l > 0
        h_ris = ch.H_ris(:, k, l);       % Nris x 1
        G_l   = ch.G(:, :, l);           % Nt x Nris

        if strcmpi(cfg.ris_phase_mode, 'random')
            theta = ch.theta(:, l);      % Nris x 1 (shared across users)
        elseif strcmpi(cfg.ris_phase_mode, 'align')
            % ---- Phase alignment baseline (per-user, per-RIS) ----
            % Use direct-link MRT direction as reference:
            hdk = ch.h_d(:, k);
            denom = norm(hdk) + cfg.eps;
            w0 = hdk / denom;            % Nt x 1

            % For each RIS element n: align phase of (w0^H g_n)*h_n
            proj = (w0' * G_l).' .* h_ris;              % Nris x 1
            theta = exp(-1j * angle(proj + cfg.eps));   % Nris x 1
        else
            error('Unknown cfg.ris_phase_mode: %s', cfg.ris_phase_mode);
        end

        % ---------- NEW: scale RIS contribution ----------
        h_eff(:, k) = h_eff(:, k) + cfg.ris_gain * (G_l * (theta .* h_ris));
    end
end
end
