import torch
from typing import Iterable, Tuple, Optional, Callable, Literal

# ---------- utilities ----------
def _pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A2 = (A*A).sum(dim=1, keepdim=True)           # (na,1)
    B2 = (B*B).sum(dim=1, keepdim=True).T         # (1,nb)
    return (A2 + B2 - 2 * (A @ B.T)).clamp_min(0.)

def _rbf_kernel(A: torch.Tensor, B: torch.Tensor, gamma: float) -> torch.Tensor:
    return torch.exp(-gamma * _pairwise_sq_dists(A, B))

def _center_K_train(Ktt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Return centered K, along with row means, col means, and grand mean (for cross-centering)
    r = Ktt.mean(dim=1, keepdim=True)      # (n,1)
    c = Ktt.mean(dim=0, keepdim=True)      # (1,n)
    g = Ktt.mean()                          # scalar
    Kc = Ktt - r - c + g
    return Kc, r, c, g

def _center_K_cross(Kvt: torch.Tensor, r_train: torch.Tensor, c_train: torch.Tensor, g_train: torch.Tensor) -> torch.Tensor:
    # Center K(val, train) relative to the training set statistics (no leakage)
    m, n = Kvt.shape
    row_means_val = Kvt.mean(dim=1, keepdim=True)      # (m,1)
    col_means_train = c_train                          # (1,n)
    g = g_train
    # k_c(xv, xt) = k - mean_over_train_of_xv - mean_over_train_of_xt + grand_mean_train
    return Kvt - row_means_val - col_means_train + g

def _fit_krr_from_gram(K: torch.Tensor, y: torch.Tensor, alpha: float, jitter: float = 1e-8) -> torch.Tensor:
    n = K.shape[0]
    I = torch.eye(n, device=K.device, dtype=K.dtype)
    A = K + (alpha + jitter) * I
    L = torch.linalg.cholesky(A)
    # solves A * W = y → W
    W = torch.cholesky_solve(y, L)
    return W  # shape (n, m)

# ---------- main API ----------
def make_krr_predictor_cv(
    train_w: torch.Tensor,
    train_y: torch.Tensor,
    *,
    alphas: Iterable[float] = tuple((10.0 ** torch.linspace(-6, 2, 9)).tolist()),
    gammas: Optional[Iterable[float]] = None,  # if None, built from median heuristic around its scale
    k_folds: int = 5,
    dtype: torch.dtype = torch.float64,
    jitter: float = 1e-8,
    seed: int = 0,
) -> Tuple[Callable[[torch.Tensor], torch.Tensor], dict]:
    """
    KRR with RBF kernel + K-fold CV over (alpha, gamma), feature- & target-centering done per fold (no leakage).
    Returns (predict_fn, info_dict).

    train_w: (n,d)
    train_y: (n,) or (n,m)
    """
    X = train_w.to(dtype)
    y = train_y.to(dtype)
    if y.dim() == 1:
        y = y.unsqueeze(1)
        squeeze_out = True
    else:
        squeeze_out = False

    n = X.shape[0]
    device = X.device
    g = torch.Generator(device=device).manual_seed(seed)
    perm = torch.randperm(n, generator=g, device=device)
    folds = torch.chunk(perm, k_folds)

    # Build gamma grid if not provided (median heuristic neighborhood)
    if gammas is None:
        with torch.no_grad():
            # sample pairs to estimate median distance (cheap for large n)
            idx = perm[:min(n, 2048)]
            D2 = _pairwise_sq_dists(X[idx], X[idx])
            med = torch.median(D2[D2 > 0]).item() if (D2 > 0).any() else 1.0
            # gamma = 1/(2 * median_dist2) times a range
            base = 1.0 / (2.0 * med) if med > 0 else 1.0
            gammas = [base * (10.0 ** p) for p in [-2, -1, -0.5, 0.0, 0.5, 1.0, 2.0]]

    best = {"rmse": float("inf"), "alpha": None, "gamma": None}

    # CV loop
    for alpha in alphas:
        for gamma in gammas:
            sqerr_sum, count = 0.0, 0
            for k in range(k_folds):
                val_idx = folds[k]
                tr_idx = torch.cat([folds[i] for i in range(k_folds) if i != k])

                Xtr, Xva = X[tr_idx], X[val_idx]
                ytr, yva = y[tr_idx], y[val_idx]

                # z-score features within fold (optional but usually helps)
                mu = Xtr.mean(dim=0, keepdim=True)
                sd = Xtr.std(dim=0, keepdim=True).clamp_min(1e-12)
                Xtr_n = (Xtr - mu) / sd
                Xva_n = (Xva - mu) / sd

                # center targets within fold
                y_mean = ytr.mean(dim=0, keepdim=True)
                ytr_c = ytr - y_mean

                # Gram matrices + proper centering
                Ktt = _rbf_kernel(Xtr_n, Xtr_n, gamma)
                Ktt_c, r, c, gmean = _center_K_train(Ktt)

                Kvt = _rbf_kernel(Xva_n, Xtr_n, gamma)
                Kvt_c = _center_K_cross(Kvt, r, c, gmean)

                # fit and predict
                W = _fit_krr_from_gram(Ktt_c, ytr_c, alpha, jitter=jitter)
                yhat_va = Kvt_c @ W + y_mean  # add back the fold's target mean

                sqerr_sum += ((yhat_va - yva) ** 2).sum().item()
                count += yva.numel()

            rmse = (sqerr_sum / count) ** 0.5
            if rmse < best["rmse"]:
                best.update({"rmse": rmse, "alpha": float(alpha), "gamma": float(gamma)})

    # Refit on all data with best hyperparameters
    # full standardization / centering
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-12)
    Xn = (X - mu) / sd
    y_mean = y.mean(dim=0, keepdim=True)
    yc = y - y_mean

    K = _rbf_kernel(Xn, Xn, best["gamma"])
    Kc, r, c, gmean = _center_K_train(K)
    W = _fit_krr_from_gram(Kc, yc, best["alpha"], jitter=jitter)

    def predict(test_w: torch.Tensor) -> torch.Tensor:
        Z = test_w.to(dtype).to(device)
        Zn = (Z - mu) / sd
        Kz = _rbf_kernel(Zn, Xn, best["gamma"])
        Kz_c = _center_K_cross(Kz, r, c, gmean)
        out = Kz_c @ W + y_mean
        return out.squeeze(1) if squeeze_out else out

    info = {"best_alpha": best["alpha"], "best_gamma": best["gamma"], "cv_rmse": best["rmse"]}
    return predict, info


# ----------------------------- shared kernel utils -----------------------------
KernelName = Literal["rbf", "linear", "poly"]

def _pairwise_sq_dists(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A2 = (A*A).sum(dim=1, keepdim=True)
    B2 = (B*B).sum(dim=1, keepdim=True).T
    return (A2 + B2 - 2 * (A @ B.T)).clamp_min(0.0)

def _kernel_and_diag(
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    kernel: KernelName,
    gamma: Optional[float],
    degree: int,
    coef0: float,
    scale: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return K(A,B) and, if A is B, the diagonal of K(A,A)."""
    if kernel == "rbf":
        if gamma is None:
            gamma = 1.0 / max(1, A.shape[1])  # safe default
        K = scale * torch.exp(-gamma * _pairwise_sq_dists(A, B))
        if A.data_ptr() == B.data_ptr() and A.shape[0] == B.shape[0]:
            diag = torch.full((A.shape[0],), fill_value=scale, dtype=A.dtype, device=A.device)
        else:
            diag = None
        return K, diag

    elif kernel == "linear":
        K = scale * (A @ B.T) + (coef0 if coef0 is not None else 0.0)
        if A.data_ptr() == B.data_ptr() and A.shape[0] == B.shape[0]:
            diag = (scale * (A * A).sum(dim=1) + (coef0 if coef0 is not None else 0.0))
        else:
            diag = None
        return K, diag

    elif kernel == "poly":
        g = gamma if gamma is not None else 1.0
        K = scale * (g * (A @ B.T) + coef0)**degree
        if A.data_ptr() == B.data_ptr() and A.shape[0] == B.shape[0]:
            diag = scale * (g * (A * A).sum(dim=1) + coef0)**degree
        else:
            diag = None
        return K, diag

    else:
        raise ValueError(f"Unknown kernel: {kernel}")


# =============================== Exact GP (GPR) ===============================
def make_gp_predictor(
    train_w: torch.Tensor,
    train_y: torch.Tensor,
    *,
    noise: float = 1e-2,                 # observation noise variance σ^2
    kernel: KernelName = "rbf",
    gamma: Optional[float] = None,       # 1/(lengthscale^2 * 2) for RBF
    degree: int = 3,
    coef0: float = 1.0,
    scale: float = 1.0,                  # kernel amplitude σ_f^2
    jitter: float = 1e-8,
    dtype: torch.dtype = torch.float64,
) -> Callable[[torch.Tensor, bool, bool], torch.Tensor]:
    """
    Fit an exact Gaussian Process regressor and return a predictor.

    Shapes
    -------
    train_w: (n, d)
    train_y: (n,) or (n, m)   # multi-output supported (independent heads)

    Returned predictor:
        predict(test_w, return_var=False, include_noise=False)
        -> mean if return_var=False
        -> (mean, var) if return_var=True
           var is shape (n_test,) and is the latent f variance unless include_noise=True
    """
    X = train_w.to(dtype)
    y = train_y.to(dtype)
    if y.dim() == 1:
        y = y.unsqueeze(1)
        squeeze_out = True
    else:
        squeeze_out = False

    device = X.device
    n = X.shape[0]

    K, _ = _kernel_and_diag(X, X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)
    K_reg = K + (noise + jitter) * torch.eye(n, dtype=X.dtype, device=device)

    L = torch.linalg.cholesky(K_reg)                  # (n,n)
    alpha = torch.cholesky_solve(y, L)                # (n,m)

    def predict(test_w: torch.Tensor, return_var: bool = False, include_noise: bool = False):
        Z = test_w.to(dtype).to(device)

        K_s, _ = _kernel_and_diag(Z, X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)
        mean = K_s @ alpha                              # (n_test, m)

        if not return_var:
            return mean.squeeze(1) if squeeze_out else mean

        # v = L^{-1} K(X,Z)
        # torch.cholesky_solve expects RHS on the right, so use transpose trick
        v = torch.cholesky_solve(K_s.T, L)             # (n, n_test)
        # k_ss diagonal
        _, kss_diag = _kernel_and_diag(Z, Z, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)
        var = kss_diag - (K_s * v.T).sum(dim=1)        # (n_test,)
        if include_noise:
            var = var + noise
        return (mean.squeeze(1) if squeeze_out else mean), var.clamp_min(0.0)

    return predict


# ================= Variational Inducing-Point GP (Titsias / VFE) ==============
def make_svgp_predictor(
    train_w: torch.Tensor,
    train_y: torch.Tensor,
    *,
    m_inducing: int = 512,                # number of inducing points if not provided
    inducing_points: Optional[torch.Tensor] = None,  # (m,d) optional
    noise: float = 1e-2,                  # observation noise variance σ^2
    kernel: KernelName = "rbf",           # SVGP here is typically with RBF; others supported but RBF is best tested
    gamma: Optional[float] = None,
    degree: int = 3,
    coef0: float = 1.0,
    scale: float = 1.0,
    jitter: float = 1e-6,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> Callable[[torch.Tensor, bool, bool], torch.Tensor]:
    """
    Variational inducing-point GP using the Titsias (2009) VFE objective
    with *closed-form* optimal q(u). No gradient training required here.

    This approximates the full GP, scales as O(n m^2 + m^3).
    For very large n, pick a moderate m (e.g., 256-1024).

    train_w: (n,d)
    train_y: (n,) or (n,m)
    Returns predict(test_w, return_var=False, include_noise=False)
    """
    X = train_w.to(dtype)
    y = train_y.to(dtype)
    if y.dim() == 1:
        y = y.unsqueeze(1)
        squeeze_out = True
    else:
        squeeze_out = False

    device = X.device
    n, d = X.shape

    # choose / set inducing points Z
    if inducing_points is None:
        g = torch.Generator(device=device).manual_seed(seed)
        m = min(m_inducing, n)
        idx = torch.randperm(n, generator=g, device=device)[:m]
        Z = X[idx].clone()
    else:
        Z = inducing_points.to(dtype).to(device)
        m = Z.shape[0]

    # Build kernel blocks
    Kuu, _ = _kernel_and_diag(Z, Z, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)
    Kuu = Kuu + jitter * torch.eye(m, dtype=dtype, device=device)           # (m,m)
    Kuf, _ = _kernel_and_diag(Z, X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)  # (m,n)

    # A = Kuu + (1/σ^2) Kuf Kuf^T
    s2 = noise
    A = Kuu + (1.0 / s2) * (Kuf @ Kuf.T)                                    # (m,m)
    LA = torch.linalg.cholesky(A)

    # μ_u = Kuu A^{-1} (1/σ^2) Kuf y
    rhs = (1.0 / s2) * (Kuf @ y)                                            # (m, out)
    tmp = torch.cholesky_solve(rhs, LA)                                     # A^{-1} (Kuf y / s2)
    mu_u = Kuu @ tmp                                                        # (m, out)

    # Σ_u = Kuu A^{-1} Kuu
    tmp2 = torch.cholesky_solve(Kuu, LA)                                    # A^{-1} Kuu
    Sigma_u = Kuu @ tmp2                                                    # (m,m)

    # Precompute Kuu^{-1} pieces for prediction
    Luu = torch.linalg.cholesky(Kuu)
    def _solve_Kuu(V: torch.Tensor) -> torch.Tensor:
        return torch.cholesky_solve(V, Luu)

    Kuu_inv_mu_u = _solve_Kuu(mu_u)                                         # (m, out)
    Mmat = Kuu - Sigma_u                                                    # (m,m)

    def predict(test_w: torch.Tensor, return_var: bool = False, include_noise: bool = False):
        Zt = test_w.to(dtype).to(device)

        Ksu, _ = _kernel_and_diag(Zt, Z, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)  # (n*, m)
        mean = Ksu @ Kuu_inv_mu_u                                                                                # (n*, out)

        if not return_var:
            return mean.squeeze(1) if squeeze_out else mean

        # var = k_ss - K_*u Kuu^{-1} (Kuu - Σ_u) Kuu^{-1} K_u*
        Kus = Ksu.T                                                                                              # (m, n*)
        A1 = _solve_Kuu(Kus)                                                                                     # (m, n*)
        A2 = Mmat @ A1                                                                                           # (m, n*)
        A3 = _solve_Kuu(A2)                                                                                      # (m, n*)
        quad_diag = (Ksu * A3.T).sum(dim=1)                                                                      # (n*,)

        _, kss_diag = _kernel_and_diag(Zt, Zt, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0, scale=scale)
        var = kss_diag - quad_diag
        if include_noise:
            var = var + s2
        return (mean.squeeze(1) if squeeze_out else mean), var.clamp_min(0.0)

    return predict