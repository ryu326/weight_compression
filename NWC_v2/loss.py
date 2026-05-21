"""Rate-distortion loss for NWC_v2 — clone of NWC.RateDistortionLoss.

    loss = lmbda * MSE(w, x_hat) / std^2 + bpp
    bpp  = sum(log2(1 / likelihoods)) / num_pixels
         = -log(likelihoods).sum() / (log(2) * num_pixels)
"""
import math

import torch
import torch.nn as nn


class RDLoss(nn.Module):
    def __init__(self, std: float, lmbda: float):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lmbda = float(lmbda)
        self.std = float(std)

    def forward(self, data, output):
        w = data["weight_block"].reshape(output["x_hat"].shape)
        num_pixels = w.numel()
        out = {}
        out["recon_loss"] = self.mse(w, output["x_hat"]) / (self.std ** 2)
        out["bpp_loss"] = (
            torch.log(output["likelihoods"]).sum() / (-math.log(2) * num_pixels)
        )
        out["loss"] = self.lmbda * out["recon_loss"] + out["bpp_loss"]
        return out
