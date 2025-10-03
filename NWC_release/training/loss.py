import torch
import torch.nn as nn
import math

class ElementwiseNormalizedMSELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(ElementwiseNormalizedMSELoss, self).__init__()
        self.epsilon = epsilon  # 0으로 나누는 것을 방지하기 위한 작은 값

    def forward(self, y_true, y_pred):
        mse_per_element = (y_true - y_pred) ** 2 / (y_true**2 + self.epsilon)
        nmse = torch.mean(mse_per_element)
        return nmse


class NormalizedMSELoss(nn.Module):
    def __init__(self, std=None):
        super(NormalizedMSELoss, self).__init__()
        self.std = std
        self.mse_fn = nn.MSELoss()

    def forward(self, y_true, y_pred, dummy=None):
        # mse = torch.mean((y_true - y_pred) ** 2)
        mse = self.mse_fn(y_true, y_pred)
        if self.std is not None:
            var = self.std**2
        else:
            var = torch.mean(y_true**2)
        return mse / var

class CalibMagScaledMSELoss(nn.Module):
    def __init__(self, std=None):
        super(CalibMagScaledMSELoss, self).__init__()
        self.std = std

    def forward(self, y_true, y_pred, scale):
        # import ipdb; ipdb.set_trace()
        ## v3
        scale = torch.clamp(scale / scale.mean(-1, keepdim=True), min=0, max = 10)
        mse_per_element = ((y_true - y_pred) ** 2) * scale 
        mse = torch.mean(mse_per_element)
        if self.std is not None:
            var = self.std**2
        else:
            raise
            var = torch.mean(y_true**2)
        return mse / var

class VQVAE_loss(nn.Module):
    def __init__(self, mse_fn):
        super(VQVAE_loss, self).__init__()
        self.mse_fn = mse_fn        

    def forward(self, data, output):
        recon_loss = self.mse_fn(output["x"], output["x_hat"], data['input_block'])
        embedding_loss = output["embedding_loss"]
        loss = recon_loss + embedding_loss
        
        return {'loss': loss,
                'recon_loss': recon_loss,
                'embedding_loss': embedding_loss}

# class RateDistortionLoss_ql(nn.Module):
#     """Custom rate distortion loss with a Lagrangian parameter."""

#     def __init__(self, std, coff, lmbda=1e-2):
#         super().__init__()
#         self.mse = nn.MSELoss()
#         self.lmbda = lmbda
#         self.std = std
#         self.coff = coff

#     def forward(self, data, output):
#         out = {}
#         # shape = output["x"].size()
#         num_pixels = output["x"].numel()

#         out["recon_loss"] = self.mse(output["x"], output["x_hat"]) / self.std**2
        
#         qlevel = data['q_level']

#         # assert output["likelihoods"].shape == self.coff[qlevel].shape
#         # print(output["likelihoods"].shape, self.coff[qlevel].shape)
        
#         ## v1
#         # out["bpp_loss"] = torch.log(output["likelihoods"]) * self.coff[qlevel].unsqueeze(-1)
#         # out["bpp_loss"] = out["bpp_loss"].sum() / (-math.log(2) * num_pixels)

#         ## v2
#         out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)
#         coff = self.coff[qlevel].reshape(output["likelihoods"].shape[0], 1, 1)
#         assert output["likelihoods"].dim() == coff.dim(), \
#             f"Shape mismatch: likelihoods {output['likelihoods'].shape} vs coff {coff.shape}"
#         bpp_loss = torch.log(output["likelihoods"]) * coff
#         bpp_loss = bpp_loss.sum() / (-math.log(2) * num_pixels)
        
#         ## v1
#         # out["loss"] = self.lmbda * out["recon_loss"] + out["bpp_loss"]
#         ## v2
#         out["loss"] = self.lmbda * out["recon_loss"] + bpp_loss
#         return out
 
class RateDistortionLoss_ql(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, coff, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std
        self.coff = coff

    def forward(self, data, output):
        out = {}
        w = data['weight_block'].reshape(output["x_hat"].shape)
        num_pixels = w.numel() 
        qlevel = data['q_level']

        coff = self.coff[qlevel].reshape(output["likelihoods"].shape[0], 1, 1)
        assert output["likelihoods"].dim() == coff.dim(), \
            f"Shape mismatch: likelihoods {output['likelihoods'].shape} vs coff {coff.shape}"
        out["recon_loss"] = self.mse(w, output["x_hat"]) / self.std**2

        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)
        bpp_loss = torch.log(output["likelihoods"]) * coff
        bpp_loss = bpp_loss.sum() / (-math.log(2) * num_pixels)
        
        out["loss"] = self.lmbda * out["recon_loss"] + bpp_loss
        return out

class RateDistortionLoss_ql_v2(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, coff, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.lmbda = lmbda
        self.std = std
        self.coff = coff

    def forward(self, data, output):
        out = {}
        w = data['weight_block'].reshape(output["x_hat"].shape)
        num_pixels = w.numel() 
        qlevel = data['q_level']

        coff = self.coff[qlevel].reshape(output["likelihoods"].shape[0], 1, 1)
        
        mse_val = self.mse(w, output["x_hat"]) 
        assert mse_val.dim() == coff.dim(), \
            f"Shape mismatch: likelihoods {mse_val.shape} vs coff {coff.shape}"
        
        out["recon_loss"] = mse_val.mean() / self.std**2

        weighted_mse = (1 / coff) * mse_val
        
        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)

        out["loss"] = self.lmbda * weighted_mse.mean() / self.std**2 + out["bpp_loss"]
        return out


class RateDistortionLoss_ql_hyper(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, coff, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std
        self.coff = coff

    def forward(self, data, output):
        out = {}
        w = data['weight_block'].reshape(output["x_hat"].shape)            
        num_pixels = w.numel() 
        qlevel = data['q_level']

        coff = self.coff[qlevel].reshape(w.shape[0], 1, 1)
        
        out["recon_loss"] = self.mse(w, output["x_hat"]) / self.std**2

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        bpp_loss = 0
        for likelihoods in output["likelihoods"].values():
            bl = torch.log(likelihoods) * coff
            bl = bl.sum() / (-math.log(2) * num_pixels)
            bpp_loss += bl
        
        out["loss"] = self.lmbda * out["recon_loss"] + bpp_loss
        return out

class RateDistortionLoss_ql_code_opt(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, coff, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss(reduction='None')
        self.lmbda = lmbda
        self.std = std
        self.coff = coff

    def forward(self, data, output, **kwargs):
        out = {}
        w = data['weight_block'].reshape(output["x_hat"].shape)
        num_pixels = w.numel() 
        qlevel = data['q_level']
        rnorm = kwargs.get('rnorm', None) # (1, m)
        cnorm = kwargs.get('cnorm', None) # (n, 1)

        coff = self.coff[qlevel].reshape(output["likelihoods"].shape[0], 1, 1)
        
        if rnorm is not None:
            out["recon_loss"] = self.mse(w, output["x_hat"]).reshape(w.shape[0], -1)
            out["recon_loss"] = (out["recon_loss"] / rnorm**2).sum()
        elif cnorm is not None:
            out["recon_loss"] = self.mse(w, output["x_hat"]).reshape(w.shape[0], -1)
            out["recon_loss"] = (out["recon_loss"] / cnorm**2).sum()
        else:
            out["recon_loss"] = self.mse(w, output["x_hat"]).sum() / self.std**2

        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)
        bpp_loss = torch.log(output["likelihoods"]) * coff
        bpp_loss = bpp_loss.sum() / (-math.log(2) * num_pixels)
        
        out["loss"] = self.lmbda * out["recon_loss"] + bpp_loss
        return out

class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, lmbda=1e-2, args = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std
        self.args = args

    def forward(self, data, output):
        out = {}

        # shape = output["x"].size()
        # num_pixels = output["x"].numel()
        # if 'scale_cond' in output.keys() and self.args.pre_normalize:
        #     output["x_hat"] = output["x_hat"] * output["scale_cond"]
        
        w = data['weight_block'].reshape(output["x_hat"].shape)
        num_pixels = w.numel() 

        out["recon_loss"] = self.mse(w, output["x_hat"]) / self.std**2
        # BPP
        # out["bpp_loss"] = sum(
        #     (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        #     for likelihoods in output["likelihoods"].values()
        # )
        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)

        # 전체 loss
        out["loss"] = self.lmbda * out["recon_loss"] + out["bpp_loss"]
        return out

class RateDistortionLoss_hyper(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std

    def forward(self, data, output):
        out = {}
        num_pixels = output["x"].numel()

        # if 'scale_cond' in output.keys() and self.args.pre_normalize:
        #     output["x_hat"] = output["x_hat"] * output["scale_cond"]

        out["recon_loss"] = self.mse(output["x"], output["x_hat"]) / self.std**2
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        out["loss"] = self.lmbda * out["recon_loss"] + out["bpp_loss"]
        return out

class ProxyHessLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, lmbda=1e-2):
        super().__init__()
        # self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std

    def forward(self, data, output):
        out = {}
        B = output["x"].size(0)
        ## (B, 10, 128, 16) --> (B, 10, 2048)
        h = data['hesseigen'].contiguous().flatten(start_dim=-2)
        num_pixels = output["x"].numel()

        ## (B, 128, 16) --> (B, 2048) --> (B, 1, 2048)
        d = (output["x"] - output["x_hat"]).contiguous().view(B, 1, -1)
        out["recon_loss"] = (d @ h.transpose(-1, -2) @ h @ d.transpose(-1, -2)).mean(dim=0) / self.std**2

        # BPP
        # import ipdb; ipdb.set_trace()
        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        # out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)

        # 전체 loss
        out["loss"] = self.lmbda * out["recon_loss"] + out["bpp_loss"]
        return out

 
class RateDistortionLoss_qmap(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, ld_min = 50, ld_max = 10000):
        super().__init__()
        self.mse = nn.MSELoss(reduction ='none')
        self.std = std
        self.min = ld_min 
        self.max = ld_max

    def lambda_from_q_tensor(self, q, lambda_min, lambda_max):
        """
        q: torch.Tensor, values in [0, 1]
        lambda_min, lambda_max: float, desired range for lambda
        returns: torch.Tensor of lambda values
        """
        log_min = torch.log(torch.tensor(lambda_min))
        log_max = torch.log(torch.tensor(lambda_max))
        log_lambda = log_min + (log_max - log_min) * q  # linear in log space
        return torch.exp(log_lambda)
    
    def forward(self, data, output):
        out = {}
        w = data['weight_block']
        
        w = w.reshape(w.shape[0], -1) ## (B, -1)
        w_hat = output['x_hat'].reshape(w.shape[0], -1) ## (B, -1)
        
        num_pixels = w.numel() 
        qmap = data['qmap']  ## (B, 1)
        
        lmbda = self.lambda_from_q_tensor(qmap, self.min, self.max) 
        
        recon_loss = self.mse(w, w_hat) / self.std**2  ## (B, -1)
        out["recon_loss"] = self.mse(w, w_hat).mean() / self.std**2 ## only for logging

        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)

        out["loss"] = (lmbda * recon_loss).mean() + out["bpp_loss"]
        return out 
    
class RateDistortionLoss_qmap_v2(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, lmbda, ld_min = 3.4, ld_max = 0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.std = std
        self.min = ld_min 
        self.max = ld_max
        self.lmbda = lmbda

    def lambda_from_q_tensor(self, q, lambda_min, lambda_max):
        """
        q: torch.Tensor, values in [0, 1]
        lambda_min, lambda_max: float, desired range for lambda
        returns: torch.Tensor of lambda values
        """
        log_min = torch.log(torch.tensor(lambda_min))
        log_max = torch.log(torch.tensor(lambda_max))
        log_lambda = log_min + (log_max - log_min) * q  # linear in log space
        return torch.exp(log_lambda)
    
    def forward(self, data, output):
        out = {}
        w = data['weight_block']
        
        w = w.reshape(w.shape[0], -1) ## (B, -1)
        w_hat = output['x_hat'].reshape(w.shape[0], -1) ## (B, -1)
        
        num_pixels = w.numel() 
        qmap = data['qmap']  ## (B, 1)
        
        lmbda_r = self.lambda_from_q_tensor(qmap, self.min, self.max) 
        
        out["recon_loss"] = self.mse(w, w_hat) / self.std**2  ## (B, -1)

        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)
        bpp_loss = torch.log(output["likelihoods"]) * lmbda_r.reshape(output["likelihoods"].shape[0], 1, 1)
        bpp_loss = bpp_loss.sum() / (-math.log(2) * num_pixels)

        out["loss"] = self.lmbda * out["recon_loss"] + bpp_loss
        return out 

def get_loss_fn(args, std=None, device = None):
    if hasattr(args, 'code_optim_lr'):
        args.lmbda = args.code_optim_lmbda
    if args.loss == "nmse":
        return VQVAE_loss(NormalizedMSELoss(std))
    elif args.loss == "enmse":
        return VQVAE_loss(ElementwiseNormalizedMSELoss())
    elif args.loss == "smse":
        return VQVAE_loss(CalibMagScaledMSELoss(std))
    elif args.loss == "rdloss":
        if args.use_hyper:
            return RateDistortionLoss_hyper(std, lmbda= args.lmbda, args = args)
        else :
            return RateDistortionLoss(std, lmbda= args.lmbda, args = args)
    elif args.loss == "rdloss_ql":
        # assert 'clip' in args.dataset_path.lower()
        if args.Q == 2:
            coff = torch.tensor([3.4, 0.05]).to(device)
        if args.Q == 4:
            coff = torch.tensor([3.4, 1.2, 0.1, 0.05]).to(device)
        if args.Q == 8:
            coff = torch.tensor([4.000, 1.707, 0.724, 0.307, 0.130, 0.055, 0.024, 0.010]).to(device)
        if args.Q == 16:
            coff = torch.tensor([4.000, 2.301, 1.324, 0.761, 0.438, 0.252, 0.145, 0.083, 0.048, 0.028, 0.016, 0.009, 0.005, 0.003, 0.0017, 0.001]).to(device)
        if args.use_hyper:
            return RateDistortionLoss_ql_hyper(std, coff, lmbda = args.lmbda)
        return RateDistortionLoss_ql(std, coff, lmbda = args.lmbda)
    elif args.loss == "rdloss_ql_v2":
        if args.Q == 2:
            coff = torch.tensor([3.4, 0.05]).to(device)
        if args.Q == 4:
            coff = torch.tensor([3.4, 1.2, 0.1, 0.05]).to(device)
        if args.Q == 8:
            coff = torch.tensor([4.000, 1.707, 0.724, 0.307, 0.130, 0.055, 0.024, 0.010]).to(device)
        if args.Q == 16:
            coff = torch.tensor([4.000, 2.301, 1.324, 0.761, 0.438, 0.252, 0.145, 0.083, 0.048, 0.028, 0.016, 0.009, 0.005, 0.003, 0.0017, 0.001]).to(device)
        assert not args.use_hyper
        
        return RateDistortionLoss_ql_v2(std, coff, lmbda = args.lmbda)
    
    elif args.loss == "rdloss_ql_code_opt":
        if args.Q == 4:
            coff = torch.tensor([3.4, 1.2, 0.1, 0.05]).to(device)
        return RateDistortionLoss_ql_code_opt(std, coff, lmbda = args.lmbda)
    elif args.loss == "proxy_hess":
        return ProxyHessLoss(std, lmbda = args.lmbda)
    elif args.loss == "rdloss_qmap":
        return RateDistortionLoss_qmap(std, args.lmbda_min, args.lmbda_max)
    elif args.loss == "rdloss_qmap2":
        return RateDistortionLoss_qmap_v2(std, lmbda = args.lmbda)
        