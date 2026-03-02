function [theta_new, info] = ris_phase_mm(cfg, ch, assign, V, weights, max_iter, theta_init)
%RIS_PHASE_MM RIS phase update via UQP + MM inner iterations.
% Paper mapping (Sec. 3.3, concise):
% 1) For RIS-l, write hk = hd_k + M_kl*theta_l under fixed (assign, V).
% 2) Build proxy objective as weighted useful-signal power (common + own private).
% 3) This gives a unit-modulus quadratic program: maximize theta^H Q theta + 2Re{q^H theta}.
% 4) Equivalently minimize f(theta)=theta^H U theta - 2Re{v^H theta}, U=-Q, v=q.
% 5) MM majorization with lambda=max eig(U):
%    theta^{t+1} = exp(j*angle((lambda*I-U)*theta^t + v)).
% 6) If non-monotone, apply damping/rollback to preserve non-worsening proxy objective.
% 7) Unit-modulus is enforced every inner iteration by projection exp(j*angle(.)) elementwise.
% 8) Under debug, per-iteration proxy objective is printed for monotonic verification.

num_ris = cfg.num_ris;
n_ris = cfg.n_ris;
K = cfg.num_users;

if nargin < 6 || isempty(max_iter)
    max_iter = get_cfg(cfg, 'ris_mm_iter', 6);
end
if nargin < 7
    theta_init = [];
end

mm_tol = get_cfg(cfg, 'ris_mm_tol', 1e-9);
alpha0 = get_cfg(cfg, 'ris_mm_alpha', 1.0);
alpha_min = get_cfg(cfg, 'ris_mm_alpha_min', 1/64);
private_scale = get_cfg(cfg, 'ris_mm_private_scale', 1.0);
debug_mm = logical(get_cfg(cfg, 'ris_mm_debug', false));

if isempty(weights)
    weights = ones(K, 1);
end
weights = weights(:);
if numel(weights) ~= K
    error('ris_phase_mm:weights_size', 'weights must be Kx1.');
end

if ~isempty(theta_init)
    theta_new = theta_init;
else
    theta_new = ch.theta;
end

if size(theta_new, 1) ~= n_ris || size(theta_new, 2) ~= num_ris
    theta_new = ch.theta;
end

affect_users = cell(num_ris, 1);
obj_history_by_ris = cell(num_ris, 1);
mono_fail_by_ris = zeros(num_ris, 1);

for l = 1:num_ris
    users_l = find(assign(:) == l);
    affect_users{l} = users_l;
    if isempty(users_l)
        obj_history_by_ris{l} = [];
        continue;
    end

    [U, v] = build_uqp_terms_for_ris(cfg, ch, V, weights, users_l, l, private_scale);

    theta_l = theta_new(:, l);
    obj_hist = zeros(max_iter + 1, 1);
    obj_prev = proxy_objective(theta_l, U, v);
    obj_hist(1) = obj_prev;

    lambda_mm = max(real(eig((U + U') * 0.5)));
    if ~isfinite(lambda_mm)
        lambda_mm = 0;
    end

    for it = 1:max_iter
        grad_vec = (lambda_mm * eye(n_ris) - U) * theta_l + v;
        theta_proj = exp(1j * angle(grad_vec + cfg.eps));

        obj_cand = proxy_objective(theta_proj, U, v);
        if obj_cand <= obj_prev + mm_tol
            theta_try = theta_proj;
            obj_try = obj_cand;
            accepted = true;
        else
            accepted = false;
            alpha = min(1, max(alpha0, alpha_min));
            theta_try = theta_l;
            obj_try = obj_prev;
            while alpha >= alpha_min
                theta_damped = exp(1j * angle((1 - alpha) * theta_l + alpha * theta_proj));
                obj_damped = proxy_objective(theta_damped, U, v);
                if obj_damped <= obj_prev + mm_tol
                    theta_try = theta_damped;
                    obj_try = obj_damped;
                    accepted = true;
                    break;
                end
                alpha = alpha * 0.5;
            end
        end

        if ~accepted
            mono_fail_by_ris(l) = mono_fail_by_ris(l) + 1;
            obj_hist(it + 1) = obj_prev;
            if debug_mm
                fprintf('[ris-mm][l=%d][it=%d] no-accept keep obj=%.6e\n', l, it, obj_prev);
            end
            break;
        end

        theta_l = theta_try;
        obj_hist(it + 1) = obj_try;
        if debug_mm
            fprintf('[ris-mm][l=%d][it=%d] obj=%.6e delta=%.3e\n', l, it, obj_try, obj_try - obj_prev);
        end

        if abs(obj_prev - obj_try) <= mm_tol
            break;
        end
        obj_prev = obj_try;
    end

    last_nz = find(obj_hist ~= 0, 1, 'last');
    if isempty(last_nz)
        last_nz = 1;
    end
    obj_history_by_ris{l} = obj_hist(1:last_nz);
    theta_new(:, l) = exp(1j * angle(theta_l + cfg.eps));
end

info = struct();
info.users_by_ris = affect_users;
info.obj_history_by_ris = obj_history_by_ris;
info.monotone_fail_count_by_ris = mono_fail_by_ris;
end

function [U, v] = build_uqp_terms_for_ris(cfg, ch, V, weights, users_l, l, private_scale)
N = cfg.n_ris;
U = zeros(N, N);
v = zeros(N, 1);

for ii = 1:numel(users_l)
    k = users_l(ii);
    wk = weights(k);

    M_kl = cfg.ris_gain * ch.G(:, :, l) * diag(ch.H_ris(:, k, l));
    h0 = ch.h_d(:, k);

    bc = M_kl' * V.v_c;
    cc = h0' * V.v_c;
    U = U - wk * (bc * bc');
    v = v + wk * conj(cc) * bc;

    vk = V.V_p(:, k);
    bp = M_kl' * vk;
    cp = h0' * vk;
    U = U - wk * private_scale * (bp * bp');
    v = v + wk * private_scale * conj(cp) * bp;
end

U = (U + U') * 0.5;
end

function obj = proxy_objective(theta, U, v)
obj = real(theta' * U * theta - 2 * real(v' * theta));
end

function val = get_cfg(cfg, name, default_val)
if isfield(cfg, name) && ~isempty(cfg.(name))
    val = cfg.(name);
else
    val = default_val;
end
end
