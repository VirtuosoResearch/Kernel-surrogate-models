import torch
import gpytorch
from typing import Callable, Optional, Literal, Tuple
from tqdm import tqdm

# ------------------------- preprocessing helpers -------------------------
def _standardize_fit(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X.mean(dim=0, keepdim=True)
    sd = X.std(dim=0, keepdim=True).clamp_min(1e-12)
    return mu, sd

def _standardize_apply(X: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (X - mu) / sd

def _center_fit(y: torch.Tensor) -> torch.Tensor:
    return y.mean(dim=0, keepdim=True)

def _center_apply(y: torch.Tensor, y_mean: torch.Tensor) -> torch.Tensor:
    return y - y_mean

def _ensure_2d_y(y: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if y.dim() == 1:
        return y.unsqueeze(1), True
    return y, False

def _gamma_to_lengthscale(gamma: Optional[float]) -> Optional[float]:
    # exp(-gamma * ||x-x'||^2) == exp(-||x-x'||^2 / (2 l^2))  →  l = sqrt(1/(2*gamma))
    if gamma is None or gamma <= 0:
        return None
    return (1.0 / (2.0 * gamma)) ** 0.5

# --------------------------- Exact GP (gpytorch) ---------------------------
class _ExactHead(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, init_lengthscale=None, init_outputscale=1.0):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # sensible inits
        if init_lengthscale is not None:
            self.covar_module.base_kernel.lengthscale = init_lengthscale
        self.covar_module.outputscale = init_outputscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def make_gpytorch_predictor(
    train_w: torch.Tensor,
    train_y: torch.Tensor,
    *,
    standardize_X: bool = True,
    center_y: bool = True,
    gamma: Optional[float] = None,       # if set, used to initialize RBF lengthscale
    init_outputscale: float = 1.0,
    init_noise: float = 1e-2,            # GaussianLikelihood noise prior init
    training_iters: int = 200,
    lr: float = 0.1,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Callable[[torch.Tensor, bool, bool], torch.Tensor]:
    """
    Exact GP via gpytorch. Trains hyperparams by maximizing the exact MLL (Adam).
    Returns: predict(test_w, return_var=False, include_noise=False)
    """
    device = device or train_w.device
    X = train_w.to(device=device, dtype=dtype).contiguous()
    y_raw = train_y.to(device=device, dtype=dtype).contiguous()
    y, squeeze_out = _ensure_2d_y(y_raw)

    # preprocessing
    muX = torch.zeros(1, X.shape[1], device=device, dtype=dtype)
    sdX = torch.ones(1, X.shape[1], device=device, dtype=dtype)
    if standardize_X:
        muX, sdX = _standardize_fit(X)
        Xn = _standardize_apply(X, muX, sdX)
    else:
        Xn = X

    y_mean = torch.zeros(1, y.shape[1], device=device, dtype=dtype)
    if center_y:
        y_mean = _center_fit(y)
        yc = _center_apply(y, y_mean)
    else:
        yc = y

    torch.manual_seed(seed)

    # train one independent GP per output head
    lengthscale_init = _gamma_to_lengthscale(gamma)
    heads, likes, opt_params = [], [], []
    for j in range(yc.shape[1]):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # init likelihood noise
        likelihood.noise = init_noise
        model = _ExactHead(Xn, yc[:, j], likelihood,
                           init_lengthscale=lengthscale_init,
                           init_outputscale=init_outputscale)
        model.to(device).to(dtype)
        likelihood.to(device).to(dtype)
        heads.append(model)
        likes.append(likelihood)
        opt_params += list(model.parameters()) + list(likelihood.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=lr)

    mlls = [gpytorch.mlls.ExactMarginalLogLikelihood(likes[j], heads[j]) for j in range(len(heads))]

    # training loop
    for it in tqdm(range(training_iters)):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for j in range(len(heads)):
            heads[j].train(); likes[j].train()
            output = heads[j](Xn)
            loss = -mlls[j](output, yc[:, j])
            total_loss = total_loss + loss
        total_loss.backward()
        optimizer.step()

    for j in range(len(heads)):
        heads[j].eval(); likes[j].eval()

    def predict(test_w: torch.Tensor, return_var: bool = False, include_noise: bool = False):
        Z = test_w.to(device=device, dtype=dtype).contiguous()
        Zn = _standardize_apply(Z, muX, sdX) if standardize_X else Z
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            means, vars_ = [], []
            for j in range(len(heads)):
                if return_var:
                    if include_noise:
                        pred = likes[j](heads[j](Zn))           # predictive (includes noise)
                        means.append(pred.mean)
                        vars_.append(pred.variance)
                    else:
                        post = heads[j](Zn)                      # latent f
                        means.append(post.mean)
                        vars_.append(post.variance)
                else:
                    pred = likes[j](heads[j](Zn))
                    means.append(pred.mean)
            mean = torch.stack(means, dim=1).squeeze(2) if means[0].dim() == 2 else torch.stack(means, dim=1)
            mean = mean + y_mean  # uncenter
            if not return_var:
                return mean.squeeze(1) if squeeze_out else mean
            var = torch.stack(vars_, dim=1).squeeze(2) if vars_[0].dim() == 2 else torch.stack(vars_, dim=1)
            return (mean.squeeze(1) if squeeze_out else mean), var.clamp_min(0.0)

    return predict

# ------------------------- SVGP (inducing points) -------------------------
class _SVGPHead(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, init_lengthscale=None, init_outputscale=1.0):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        if init_lengthscale is not None:
            self.covar_module.base_kernel.lengthscale = init_lengthscale
        self.covar_module.outputscale = init_outputscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def make_gpytorch_svgp_predictor(
    train_w: torch.Tensor,
    train_y: torch.Tensor,
    *,
    m_inducing: int = 512,
    inducing_points: Optional[torch.Tensor] = None,
    standardize_X: bool = True,
    center_y: bool = True,
    gamma: Optional[float] = None,
    init_outputscale: float = 1.0,
    init_noise: float = 1e-2,
    training_iters: int = 500,
    lr: float = 0.01,
    batch_size: Optional[int] = None,   # if None → full batch
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> Callable[[torch.Tensor, bool, bool], torch.Tensor]:
    """
    Sparse variational GP via gpytorch (inducing points).
    Returns: predict(test_w, return_var=False, include_noise=False)
    """
    device = device or train_w.device
    X = train_w.to(device=device, dtype=dtype).contiguous()
    y_raw = train_y.to(device=device, dtype=dtype).contiguous()
    y, squeeze_out = _ensure_2d_y(y_raw)

    muX = torch.zeros(1, X.shape[1], device=device, dtype=dtype)
    sdX = torch.ones(1, X.shape[1], device=device, dtype=dtype)
    if standardize_X:
        muX, sdX = _standardize_fit(X)
        Xn = _standardize_apply(X, muX, sdX)
    else:
        Xn = X

    y_mean = torch.zeros(1, y.shape[1], device=device, dtype=dtype)
    if center_y:
        y_mean = _center_fit(y)
        yc = _center_apply(y, y_mean)
    else:
        yc = y

    torch.manual_seed(seed)

    n, d = Xn.shape
    lengthscale_init = _gamma_to_lengthscale(gamma)

    # choose inducing points
    if inducing_points is None:
        m = min(m_inducing, n)
        idx = torch.randperm(n, device=device)[:m]
        Z = Xn[idx].clone()
    else:
        Z = inducing_points.to(device=device, dtype=dtype)
        m = Z.shape[0]

    heads, likes, opt_params = [], [], []
    for j in range(yc.shape[1]):
        model = _SVGPHead(Z.clone(), init_lengthscale=lengthscale_init, init_outputscale=init_outputscale).to(device).to(dtype)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = init_noise
        likelihood.to(device).to(dtype)
        heads.append(model)
        likes.append(likelihood)
        opt_params += list(model.parameters()) + list(likelihood.parameters())

    optimizer = torch.optim.Adam(opt_params, lr=lr)

    # ELBOs (note: num_data = n for each head)
    elbos = [gpytorch.mlls.VariationalELBO(likes[j], heads[j], num_data=n) for j in range(len(heads))]

    # minibatching
    if batch_size is None or batch_size >= n:
        batches = [(Xn, yc)]
    else:
        perm = torch.randperm(n, device=device)
        def batch_iter():
            for start in range(0, n, batch_size):
                idx = perm[start:start+batch_size]
                yield Xn[idx], yc[idx]
        batches = list(batch_iter())

    # training loop
    for it in range(training_iters):
        # shuffle each epoch if minibatching
        if batch_size is not None and batch_size < n:
            perm = torch.randperm(n, device=device)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        if batch_size is None or batch_size >= n:
            batch_iterable = batches
        else:
            batch_iterable = []
            for start in range(0, n, batch_size):
                idx = perm[start:start+batch_size]
                batch_iterable.append((Xn[idx], yc[idx]))
        for xb, yb in batch_iterable:
            loss_b = 0.0
            for j in range(len(heads)):
                heads[j].train(); likes[j].train()
                out = heads[j](xb)
                loss_b = loss_b + (-elbos[j](out, yb[:, j]))
            loss_b.backward(retain_graph=False)
            total_loss += float(loss_b.detach().cpu())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # (optional) you can print total_loss here

    for j in range(len(heads)):
        heads[j].eval(); likes[j].eval()

    def predict(test_w: torch.Tensor, return_var: bool = False, include_noise: bool = False):
        Zt = test_w.to(device=device, dtype=dtype).contiguous()
        Zn = _standardize_apply(Zt, muX, sdX) if standardize_X else Zt
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            means, vars_ = [], []
            for j in range(len(heads)):
                latent = heads[j](Zn)
                if return_var:
                    if include_noise:
                        pred = likes[j](latent)
                        means.append(pred.mean)
                        vars_.append(pred.variance)
                    else:
                        means.append(latent.mean)
                        vars_.append(latent.variance)
                else:
                    pred = likes[j](latent)
                    means.append(pred.mean)
            mean = torch.stack(means, dim=1).squeeze(2) if means[0].dim() == 2 else torch.stack(means, dim=1)
            mean = mean + y_mean
            if not return_var:
                return mean.squeeze(1) if squeeze_out else mean
            var = torch.stack(vars_, dim=1).squeeze(2) if vars_[0].dim() == 2 else torch.stack(vars_, dim=1)
            return (mean.squeeze(1) if squeeze_out else mean), var.clamp_min(0.0)

    return predict
