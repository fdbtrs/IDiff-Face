import math

import torch
from tqdm import tqdm


class DenoisingDiffusionProbabilisticModel(torch.nn.Module):
    def __init__(self,
                 eps_model: torch.nn.Module,
                 T: int,
                 criterion: torch.nn.Module = torch.nn.MSELoss(),
                 schedule_type: str = 'linear',
                 schedule_k: float = None,
                 schedule_beta_min: float = None,
                 schedule_beta_max: float = None) -> None:

        super(DenoisingDiffusionProbabilisticModel, self).__init__()

        self.eps_model = eps_model

        # register_buffer allows us to freely access these tensors by name. It helps device placement.
        betas = compute_beta_schedule(T, schedule_type,
                                      k=schedule_k, beta_min=schedule_beta_min, beta_max=schedule_beta_max)
        for k, v in precompute_schedule_constants(betas).items():
            self.register_buffer(k, v)

        self.T = T
        self.criterion = criterion
        self.schedule_type = schedule_type
        self.schedule_k = schedule_k

    def forward(self, x0: torch.Tensor, context: torch.Tensor = None, dropout_mask: torch.Tensor = None) -> torch.Tensor:
        # t ~ U(0, T)
        t = torch.randint(0, self.T, (x0.shape[0],)).to(x0.device)
        # eps ~ N(0, 1)
        eps = torch.randn_like(x0)

        # get mean and standard deviation of p(x_t|x_0)
        mean = self.sqrt_alpha_bars[t, None, None, None] * x0
        sd = self.sqrt_one_minus_alpha_bars[t, None, None, None]

        # sample from p(x_t|x_0)
        x_t = mean + sd * eps

        return self.criterion(eps, self.eps_model(x_t, t, context, dropout_mask))

    def sample(self, n_samples, size, x_T: torch.Tensor = None, context: torch.Tensor = None, dropout_mask: torch.Tensor = None) -> torch.Tensor:
        # if initial noise is not provided then sample it
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        # this samples accordingly to Algorithm 2
        self.eval()
        with torch.no_grad():

            pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            pbar.set_description("DDPM Sampling")

            for i in pbar:
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps = self.eps_model(x_t, t, context, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + self.sigmas[i] * z

        self.train()
        return x_t

    def sample_classifier_free_guided(self, n_samples, size, context, guidance_scale, x_T: torch.Tensor = None) -> torch.Tensor:
        # CLASSIFIER-FREE DIFFUSION GUIDANCE: https://arxiv.org/pdf/2207.12598.pdf

        # if initial noise is not provided then sample it
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        if type(guidance_scale) is float:
            guidance_scale = [guidance_scale] * self.T
	
        self.eval()
        with torch.no_grad():

            pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            pbar.set_description(f"DDPM Classifier-Free Guided Sampling")

            for i in pbar:
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps_unconditional = self.eps_model(x_t, t, context=None, dropout_mask=None)
                eps_conditional = self.eps_model(x_t, t, context, dropout_mask=None)

                eps = (1 - guidance_scale[i]) * eps_unconditional + guidance_scale[i] * eps_conditional

                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + \
                      self.sigmas[i] * z

        self.train()
        return x_t

    def sample_and_get_step_results(self, n_samples, size, x_T: torch.Tensor = None, context: torch.Tensor = None, dropout_mask: torch.Tensor = None, result_steps = None) -> torch.Tensor:
        # if initial noise is not provided then sample it
        x_t = x_T if x_T is not None else self.sample_prior(n_samples, size).cuda()

        result_xs = {0: x_t.clone()}

        # this samples accordingly to Algorithm 2
        self.eval()
        with torch.no_grad():

            pbar = tqdm(reversed(range(0, self.T)), total=self.T)
            pbar.set_description("DDPM Sampling")

            for i in pbar:
                z = torch.randn(n_samples, *size).cuda() if i > 1 else 0
                t = torch.tensor(i).repeat(n_samples).cuda()

                eps = self.eps_model(x_t, t, context, dropout_mask)
                x_t = self.sqrt_alphas_inv[i] * (x_t - eps * self.one_minus_alphas_over_sqrt_one_minus_alpha_bars[i]) + self.sigmas[i] * z

                if result_steps is not None and i+1 in result_steps:
                    result_xs[i] = x_t.clone()

        result_xs[self.T] = x_t

        self.train()
        return result_xs

    @staticmethod
    def sample_prior(n_samples, size):
        return torch.randn(n_samples, *size)


def compute_beta_schedule(
        T: int, schedule_type: str = 'linear', k: float = 1.0,
        beta_min: float = None, beta_max: float = None) -> torch.Tensor:

    if schedule_type.lower() == 'linear':
        scale = 1000 / T
        beta_1 = scale * 0.0001
        beta_T = scale * 0.02
        return torch.linspace(beta_1, beta_T, T, dtype=torch.float32)

    elif schedule_type.lower() in ['cosine', 'cosine_warped']:
        # custom modification to cosine schedule -> warped cosine schedule
        # (this is equivalent to original cosine schedule if k=1 and beta_min=0.0)

        s = 0.008
        beta_min = 0.0 if schedule_type.lower() == 'cosine' else beta_min
        k = 1 if schedule_type.lower() == 'cosine' else k

        return betas_for_alpha_bar(
            T, lambda t: math.cos(math.pi / 2 * (t + s) / (1 + s) ** k) ** 2,
            beta_min=beta_min, beta_max=beta_max
        )

    raise NotImplementedError


def betas_for_alpha_bar(T, alpha_bar, beta_min=0.0, beta_max=1.0):
    betas = []
    for i in range(T):
        t1 = i / T
        t2 = (i + 1) / T
        betas.append(min(max(1 - alpha_bar(t2) / alpha_bar(t1), beta_min), beta_max))
    return torch.tensor(betas).float()


def precompute_schedule_constants(betas: torch.Tensor):
    alphas = 1 - betas
    sqrt_alphas_inv = 1 / alphas.sqrt()

    sigmas = betas.sqrt()

    alpha_bars = torch.cumsum(torch.log(alphas), dim=0).exp()
    sqrt_alpha_bars = alpha_bars.sqrt()

    sqrt_one_minus_alpha_bars = (1 - alpha_bars).sqrt()
    one_minus_alphas_over_sqrt_one_minus_alpha_bars = (1 - alphas) / sqrt_one_minus_alpha_bars

    """
    import matplotlib.pyplot as plt
    plt.title("Variance Schedule")
    plt.plot(betas, label="betas")
    plt.plot(alpha_bars, label="alpha_bars")
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
    """

    return {
        "betas": betas,
        "alphas": alphas,
        "sigmas": sigmas,
        "sqrt_alphas_inv": sqrt_alphas_inv,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": sqrt_alpha_bars,
        "sqrt_one_minus_alpha_bars": sqrt_one_minus_alpha_bars,
        "one_minus_alphas_over_sqrt_one_minus_alpha_bars": one_minus_alphas_over_sqrt_one_minus_alpha_bars
    }
