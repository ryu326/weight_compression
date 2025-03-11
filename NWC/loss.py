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

        # shape = output["x"].size()
        num_pixels = output["x"].numel()

        out["recon_loss"] = self.mse(output["x"], output["x_hat"]) / self.std**2
        
        qlevel = data['q_level']
                
        # assert output["likelihoods"].shape == self.coff[qlevel].shape
        # print(output["likelihoods"].shape, self.coff[qlevel].shape)
        
        ## v1
        # out["bpp_loss"] = torch.log(output["likelihoods"]) * self.coff[qlevel].unsqueeze(-1)
        # out["bpp_loss"] = out["bpp_loss"].sum() / (-math.log(2) * num_pixels)

        ## v2
        out["bpp_loss"] = torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)
        bpp_loss = torch.log(output["likelihoods"]) * self.coff[qlevel].unsqueeze(-1).unsqueeze(-1)
        bpp_loss = bpp_loss.sum() / (-math.log(2) * num_pixels)
        
        ## v1
        # out["loss"] = self.lmbda * out["recon_loss"] + out["bpp_loss"]
        ## v2
        out["loss"] = self.lmbda * out["recon_loss"] + bpp_loss
        return out
 
class RateDistortionLoss(nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, std, lmbda=1e-2):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = lmbda
        self.std = std

    def forward(self, data, output):
        out = {}

        # shape = output["x"].size()
        num_pixels = output["x"].numel()

        out["recon_loss"] = self.mse(output["x"], output["x_hat"]) / self.std**2

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

 
def get_loss_fn(args, std=None, device = None):
    if args.loss == "nmse":
        return VQVAE_loss(NormalizedMSELoss(std))
    elif args.loss == "enmse":
        return VQVAE_loss(ElementwiseNormalizedMSELoss())
    elif args.loss == "smse":
        return VQVAE_loss(CalibMagScaledMSELoss(std))
    elif args.loss == "rdloss":
        return RateDistortionLoss(std, lmbda= args.lmbda)
    elif args.loss == "rdloss_ql":
        if args.Q == 4:
            coff = torch.tensor([3.4, 1.2, 0.1, 0.05]).to(device)
        if args.Q == 8:
            coff = torch.tensor([4.000, 1.707, 0.724, 0.307, 0.130, 0.055, 0.024, 0.010]).to(device)
        if args.Q == 16:
            coff = torch.tensor([4.000, 2.301, 1.324, 0.761, 0.438, 0.252, 0.145, 0.083, 0.048, 0.028, 0.016, 0.009, 0.005, 0.003, 0.0017, 0.001]).to(device)
        return RateDistortionLoss_ql(std, coff, lmbda = args.lmbda)
    elif args.loss == "proxy_hess":
        return ProxyHessLoss(std, lmbda = args.lmbda)