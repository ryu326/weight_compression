import torch
import torch.nn as nn
import numpy as np
from scipy.linalg import hadamard

def hadamard_transform(x):
    B, H, W = x.shape
    assert H == W and (H & (H - 1)) == 0, "Size must be power of 2"
    H_mat = torch.from_numpy(hadamard(H).astype(np.float32)).to(x.device) / np.sqrt(H)
    return H_mat @ x @ H_mat.T

class PatchwiseHadamard(nn.Module):
    def __init__(self, patch_size=256, conv_channels=16):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(1, conv_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(conv_channels, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        x = x.unfold(2, P, P).unfold(3, P, P)  # B,C,H//P,W//P,P,P
        x = x.contiguous().view(-1, P, P)      # Flatten batch-patch dimension

        x_h = hadamard_transform(x)            # Forward Hadamard
        x_h = x_h.unsqueeze(1)                 # Add channel dim

        x_h = self.conv(x_h)
        x_h = self.out_conv(x_h)

        x_out = hadamard_transform(x_h.squeeze(1))  # Inverse Hadamard
        x_out = x_out.unsqueeze(1)

        # Reassemble patches
        B_out = B
        H_p, W_p = H // P, W // P
        x_out = x_out.view(B_out, H_p, W_p, 1, P, P)
        x_out = x_out.permute(0, 3, 1, 4, 2, 5).contiguous()
        return x_out.view(B_out, 1, H, W)

class HadamardFFCDecoder(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, patch_size=256):
        super().__init__()
        self.hadamard_branch = PatchwiseHadamard(patch_size, mid_channels)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(mid_channels, in_channels, 3, padding=1)

    def forward(self, x):
        x = self.hadamard_branch(x)
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x
