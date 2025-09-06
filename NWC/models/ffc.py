import torch
import torch.nn as nn
import torch.fft

class PatchwiseFourier(nn.Module):
    def __init__(self, patch_size=256, fourier_channels=64):
        super().__init__()
        self.patch_size = patch_size
        self.fourier_conv = nn.Conv2d(2, fourier_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(fourier_channels, 2, kernel_size=1)  # to complex

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0, "H and W must be divisible by patch size"

        x = x.unfold(2, P, P).unfold(3, P, P)  # (B, C, H//P, W//P, P, P)
        x = x.contiguous().view(-1, C, P, P)  # (B * N, C, P, P)

        fft = torch.fft.rfft2(x, norm='ortho')
        fft_feat = torch.cat([fft.real, fft.imag], dim=1)
        out = self.fourier_conv(fft_feat)
        out = self.out_conv(out)
        real, imag = out.chunk(2, dim=1)
        out_complex = torch.complex(real, imag)
        out_spatial = torch.fft.irfft2(out_complex, s=(P, P), norm='ortho')

        # Reassemble patches
        B_out = B
        H_p, W_p = H // P, W // P
        out_spatial = out_spatial.view(B_out, H_p, W_p, 1, P, P)
        out_spatial = out_spatial.permute(0, 3, 1, 4, 2, 5).contiguous()
        return out_spatial.view(B_out, 1, H, W)

class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, ratio_gin=0.5, ratio_gout=0.5):
        super().__init__()
        c_l_in = int((1 - ratio_gin) * in_channels)
        c_g_in = in_channels - c_l_in
        c_l_out = int((1 - ratio_gout) * out_channels)
        c_g_out = out_channels - c_l_out

        self.local = nn.Conv2d(c_l_in, c_l_out, 3, padding=1)
        self.global_1x1 = nn.Conv2d(c_g_in, c_g_out, 1) if c_g_in > 0 else None
        self.fourier_conv = nn.Conv2d(c_g_in, c_g_out, 1) if c_g_in > 0 else None
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.split = c_l_in

    def forward(self, x):
        x_l, x_g = torch.split(x, [self.split, x.shape[1] - self.split], dim=1)
        out_l = self.local(x_l)
        if self.fourier_conv:
            out_g = self.fourier_conv(x_g) + self.global_1x1(x_g)
            out = torch.cat([out_l, out_g], dim=1)
        else:
            out = out_l
        return self.act(self.bn(out)), None

class PatchwiseFFCDecoder(nn.Module):
    def __init__(self, in_channels=1, mid_channels=16, patch_size=256):
        super().__init__()
        self.patch_fft = PatchwiseFourier(patch_size=patch_size, fourier_channels=mid_channels)
        self.ffc1 = FFC_BN_ACT(in_channels, mid_channels, ratio_gin=0.5, ratio_gout=0.5)
        self.ffc2 = FFC_BN_ACT(mid_channels, in_channels, ratio_gin=0.5, ratio_gout=0.5)

    def forward(self, x):
        x = self.patch_fft(x)
        x, _ = self.ffc1(x)
        x, _ = self.ffc2(x)
        return x
