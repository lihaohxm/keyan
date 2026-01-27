function [W, Theta, h_eff_opt] = ao_beamforming(cfg, ch, assign, p_dbw, opts)
%AO_BEAMFORMING Alternating Optimization for BS beamforming W and RIS phase Theta
%
% Given user-RIS assignment, jointly optimize:
%   - W: BS transmit beamforming matrix (Nt x K)
%   - Theta: RIS phase shifts (N x L diagonal matrices)
%
% Algorithm:
%   1. Initialize Theta (align or random)
%   2. Fix Theta, optimize W (MRT/MMSE)
%   3. Fix W, optimize Theta (closed-form phase alignment)
%   4. Repeat until convergence
%
% Inputs:
%   cfg    - configuration struct
%   ch     - channel struct (h_d, G, H_ris, theta)
%   assign - user assignment vector (K x 1), 0=direct, 1..L=RIS index
%   p_dbw  - transmit power in dBW
%   opts   - options struct
%
% Outputs:
%   W         - optimized beamforming matrix (Nt x K)
%   Theta     - optimized RIS phases (N x L)
%   h_eff_opt - effective channel after optimization (Nt x K)

if nargin < 5, opts = struct(); end

% Parameters
max_iter = get_opt(opts, 'max_iter', 10);
tol = get_opt(opts, 'tol', 1e-4);
verbose = get_opt(opts, 'verbose', false);

K = cfg.num_users;
L = cfg.num_ris;
Nt = cfg.nt;
N = cfg.n_ris;

p_watts = 10^(p_dbw / 10);
noise_power = cfg.noise_watts;

% RIS gain
ris_gain = 1;
if isfield(cfg, 'ris_gain'), ris_gain = cfg.ris_gain; end

% Initialize Theta from channel (or align)
Theta = ch.theta;  % N x L

% Track convergence
prev_sum_rate = -inf;

for iter = 1:max_iter
    
    % ========== Step 1: Compute effective channel with current Theta ==========
    h_eff = compute_effective_channel(cfg, ch, assign, Theta, ris_gain);
    
    % ========== Step 2: Fix Theta, optimize W (MRT beamforming) ==========
    % MRT: w_k = h_eff_k / ||h_eff_k||
    W = zeros(Nt, K);
    for k = 1:K
        h_k = h_eff(:, k);
        W(:, k) = h_k / (norm(h_k) + cfg.eps);
    end
    
    % Power normalization: equal power allocation
    p_k = p_watts / K;
    W = sqrt(p_k) * W;
    
    % ========== Step 3: Fix W, optimize Theta (phase alignment) ==========
    for l = 1:L
        % Find users assigned to RIS l
        users_l = find(assign == l);
        if isempty(users_l)
            continue;
        end
        
        G_l = ch.G(:, :, l);  % Nt x N
        
        % Aggregate phase optimization for all users on this RIS
        % Goal: align (w_k^H * G_l * diag(theta_l) * h_ris_k) with direct path
        
        phase_sum = zeros(N, 1);
        for k = users_l(:)'
            w_k = W(:, k);
            h_ris_k = ch.H_ris(:, k, l);  % N x 1
            h_d_k = ch.h_d(:, k);         % Nt x 1
            
            % Compute: (w_k^H * G_l)^T .* h_ris_k
            % We want theta_l to align this with h_d_k
            a = (w_k' * G_l).';  % N x 1
            b = a .* h_ris_k;    % N x 1
            
            % Desired phase: align with direct link contribution
            direct_contrib = w_k' * h_d_k;
            
            % Optimal phase for each element
            phase_sum = phase_sum + conj(b) * direct_contrib;
        end
        
        % Average and extract phase
        Theta(:, l) = exp(1j * angle(phase_sum + cfg.eps));
    end
    
    % ========== Step 4: Check convergence ==========
    h_eff = compute_effective_channel(cfg, ch, assign, Theta, ris_gain);
    [~, ~, sum_rate] = sinr_rate(cfg, h_eff, p_dbw);
    
    if verbose
        fprintf('AO iter %d: sum_rate = %.4f\n', iter, sum_rate);
    end
    
    if abs(sum_rate - prev_sum_rate) < tol
        break;
    end
    prev_sum_rate = sum_rate;
end

% Final effective channel
h_eff_opt = compute_effective_channel(cfg, ch, assign, Theta, ris_gain);

end

%% ========== Helper Functions ==========

function v = get_opt(opts, f, d)
    if isfield(opts, f), v = opts.(f); else, v = d; end
end

function h_eff = compute_effective_channel(cfg, ch, assign, Theta, ris_gain)
    K = cfg.num_users;
    h_eff = ch.h_d;  % Start with direct channel
    
    for k = 1:K
        l = assign(k);
        if l > 0
            theta_l = Theta(:, l);
            h_ris_k = ch.H_ris(:, k, l);
            G_l = ch.G(:, :, l);
            
            h_eff(:, k) = h_eff(:, k) + ris_gain * G_l * (theta_l .* h_ris_k);
        end
    end
end
