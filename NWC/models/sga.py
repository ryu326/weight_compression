import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, EntropyBottleneckVbr
# import wandb
class Quantizator_SGA(nn.Module):
    """
    https://github.com/mandt-lab/improving-inference-for-neural-image-compression/blob/c9b5c1354a38e0bb505fc34c6c8f27170f62a75b/sga.py#L110
    Stochastic Gumbeling Annealing
    sample() has no grad, so we choose STE to backward. We can also try other estimate func.
    """

    def __init__(self, gap=1000, c=0.002):
        super(Quantizator_SGA, self).__init__()
        self.gap = gap
        self.c = c

    def annealed_temperature(self, t, r, ub, lb=1e-8, backend=np, scheme='exp', **kwargs):
        """
        Return the temperature at time step t, based on a chosen annealing schedule.
        :param t: step/iteration number
        :param r: decay strength
        :param ub: maximum/init temperature
        :param lb: small const like 1e-8 to prevent numerical issue when temperature gets too close to 0
        :param backend: np or tf
        :param scheme:
        :param kwargs:
        :return:
        """
        default_t0 = kwargs.get('t0')

        if scheme == 'exp':
            tau = backend.exp(-r * t)
        elif scheme == 'exp0':
            # Modified version of above that fixes temperature at ub for initial t0 iterations
            t0 = kwargs.get('t0', default_t0)
            tau = ub * backend.exp(-r * (t - t0))
        elif scheme == 'linear':
            # Cool temperature linearly from ub after the initial t0 iterations
            t0 = kwargs.get('t0', default_t0)
            tau = -r * (t - t0) + ub
        else:
            raise NotImplementedError

        if backend is None:
            return min(max(tau, lb), ub)
        else:
            return backend.minimum(backend.maximum(tau, lb), ub)

    def forward(self, input, it=None, mode=None, total_it=None):
        if mode == "training":
            assert it is not None
            x_floor = torch.floor(input)
            x_ceil = torch.ceil(input)
            x_bds = torch.stack([x_floor, x_ceil], dim=-1)

            eps = 1e-5

            annealing_scheme = 'exp0'
            annealing_rate = 1e-3  # default annealing_rate = 1e-3
            t0 = int(total_it * 0.35)  # default t0 = 700 for 2000 iters
            T_ub = 0.5

            T = self.annealed_temperature(it, r=annealing_rate, ub=T_ub, scheme=annealing_scheme, t0=t0)

            x_interval1 = torch.clamp(input - x_floor, -1 + eps, 1 - eps)
            x_atanh1 = torch.log((1 + x_interval1) / (1 - x_interval1)) / 2
            x_interval2 = torch.clamp(x_ceil - input, -1 + eps, 1 - eps)
            x_atanh2 = torch.log((1 + x_interval2) / (1 - x_interval2)) / 2

            rx_logits = torch.stack([-x_atanh1 / T, -x_atanh2 / T], dim=-1)
            rx = F.softmax(rx_logits, dim=-1)  # just for observation in tensorboard
            rx_dist = torch.distributions.RelaxedOneHotCategorical(T, rx)

            rx_sample = rx_dist.rsample()

            x_tilde = torch.sum(x_bds * rx_sample, dim=-1)
            
            with torch.no_grad():
                x_soft = x_tilde
                x_hard = torch.round(input)
                diff = (x_soft - x_hard).abs().mean()
                wandb.log({"hard_soft_diff": diff})
            
            return x_tilde
        else:
            return torch.round(input)
        

class EntropyBottleneckNoQuant(EntropyBottleneck):
    def __init__(self, channels):
        super().__init__(channels)
        self.sga = Quantizator_SGA()

    def forward(self, x_quant):
        perm = np.arange(len(x_quant.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x_quant.shape))[np.argsort(perm)]
        x_quant = x_quant.permute(*perm).contiguous()
        shape = x_quant.size()
        x_quant = x_quant.reshape(x_quant.size(0), 1, -1)
        # likelihood = self._likelihood(x_quant)
        likelihood, _, _ = self._likelihood(x_quant)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return likelihood

class EntropyBottleneckNoQuantVbr(EntropyBottleneckVbr):
    def __init__(self, channels):
        super().__init__(channels)
        self.sga = Quantizator_SGA()

    def forward(self, x_quant, qs):
        perm = np.arange(len(x_quant.shape))
        perm[0], perm[1] = perm[1], perm[0]
        # Compute inverse permutation
        inv_perm = np.arange(len(x_quant.shape))[np.argsort(perm)]
        x_quant = x_quant.permute(*perm).contiguous()
        shape = x_quant.size()
        x_quant = x_quant.reshape(x_quant.size(0), 1, -1)
        
        if qs is None:  
            likelihood, _, _ = self._likelihood(x_quant)
        else:
            likelihood, _, _ = self._likelihood_variable(x_quant, qs = qs)
        
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        # Convert back to input tensor shape
        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()
        return likelihood