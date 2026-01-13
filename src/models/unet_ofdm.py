"""
Time-domain U-Net model for OFDM signals.

This is a 1D U-Net architecture operating directly in the time domain,
without any frequency transform (no CQT/STFT).

Based on the architecture from unet_1d.py but optimized for OFDM signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio


class CombinerUp(nn.Module):
    """Combining after upsampling in the decoder side."""

    def __init__(self, mode, Npyr, Nx, bias=True):
        super().__init__()
        self.conv1x1 = nn.Conv1d(Nx, Npyr, 1, bias=bias)
        self.mode = mode
        torch.nn.init.constant_(self.conv1x1.weight, 0)

    def forward(self, pyr, x):
        if self.mode == "sum":
            x = self.conv1x1(x)
            if pyr is None:
                return x
            else:
                return (pyr[..., 0:x.shape[-1]] + x) / (2 ** 0.5)
        else:
            raise NotImplementedError


class CombinerDown(nn.Module):
    """Combining after downsampling in the encoder side."""

    def __init__(self, mode, Nin, Nout, bias=True):
        super().__init__()
        self.conv1x1 = nn.Conv1d(Nin, Nout, 1, bias=bias)
        self.mode = mode

    def forward(self, pyr, x):
        if self.mode == "sum":
            pyr = self.conv1x1(pyr)
            return (pyr + x) / (2 ** 0.5)
        else:
            raise NotImplementedError


class Upsample(nn.Module):
    """Upsample time dimension."""

    def __init__(self, S):
        super().__init__()
        N = 2 ** 12
        self.resample = torchaudio.transforms.Resample(N, N * S)

    def forward(self, x):
        return self.resample(x)


class Downsample(nn.Module):
    """Downsample time dimension."""

    def __init__(self, S):
        super().__init__()
        N = 2 ** 12
        self.resample = torchaudio.transforms.Resample(N, N / S)

    def forward(self, x):
        return self.resample(x)


class RFF_MLP_Block(nn.Module):
    """
    Encoder of the noise level embedding.
    Random Fourier Feature embedding + MLP.
    """

    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
            sigma: (shape: [B, 1], dtype: float32)
        Returns:
            x: embedding of sigma (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Film(nn.Module):
    """Feature-wise Linear Modulation for noise conditioning."""

    def __init__(self, output_dim, bias=True):
        super().__init__()
        self.bias = bias
        if bias:
            self.output_layer = nn.Linear(512, 2 * output_dim)
        else:
            self.output_layer = nn.Linear(512, 1 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        if self.bias:
            gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        else:
            gamma = sigma_encoding
            beta = None
        return gamma, beta


class Gated_residual_layer(nn.Module):
    """Gated residual convolution layer with dilation."""

    def __init__(self, dim, kernel_size, dilation, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=1,
            padding='same',
            padding_mode='zeros',
            bias=bias
        )
        self.act = nn.GELU()

    def forward(self, x):
        x = (x + self.conv(self.act(x))) / (2 ** 0.5)
        return x


class ResnetBlock(nn.Module):
    """ResNet block with FiLM conditioning."""

    def __init__(self, dim, dim_out, use_norm=False, groups=8, bias=True):
        super().__init__()
        self.bias = bias
        self.use_norm = use_norm
        self.film = Film(dim, bias=bias)

        self.res_conv = nn.Conv1d(dim, dim_out, 1, padding_mode="zeros", bias=bias) \
            if dim != dim_out else nn.Identity()

        self.H = nn.ModuleList()
        self.num_layers = 8

        if self.use_norm:
            self.gnorm = nn.GroupNorm(8, dim)

        self.first_conv = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(dim, dim_out, 1, bias=bias)
        )

        for i in range(self.num_layers):
            self.H.append(Gated_residual_layer(dim_out, 5, 2 ** i, bias=bias))

    def forward(self, x, sigma):
        gamma, beta = self.film(sigma)

        if self.use_norm:
            x = self.gnorm(x)

        if self.bias:
            x = x * gamma + beta
        else:
            x = x * gamma

        y = self.first_conv(x)

        for h in self.H:
            y = h(y)

        return (y + self.res_conv(x)) / (2 ** 0.5)


class CropConcatBlock(nn.Module):
    """Crop and concatenate for skip connections."""

    def forward(self, down_layer, x, **kwargs):
        x1_shape = down_layer.shape
        x2_shape = x.shape

        height_diff = (x1_shape[2] - x2_shape[2]) // 2
        down_layer_cropped = down_layer[:, :, height_diff: (x2_shape[2] + height_diff)]
        x = torch.cat((down_layer_cropped, x), 1)
        return x


class Unet_OFDM(nn.Module):
    """
    Time-domain U-Net for OFDM signals.

    This model operates directly on time-domain signals without any
    frequency transform, making it suitable for OFDM declipping.
    """

    def __init__(self, args, device):
        super(Unet_OFDM, self).__init__()
        self.args = args
        self.depth = 6
        self.embedding = RFF_MLP_Block()

        # Get use_norm from config or default to False
        self.use_norm = getattr(args, 'ofdm_use_norm', False)
        if hasattr(args, 'cqt') and hasattr(args.cqt, 'use_norm'):
            self.use_norm = args.cqt.use_norm

        self.device = device
        Nin = 1  # Single channel (real-valued OFDM signal)

        # Channel dimensions for each level
        self.Ns = [64, 64, 128, 128, 256, 256, 256]
        self.Ss = [2, 2, 2, 2, 2, 2]  # Downsampling factors

        # Initial convolution
        self.init_conv = nn.Conv1d(Nin, self.Ns[0], 5, padding="same", padding_mode="zeros", bias=False)

        # Encoder, middle, and decoder modules
        self.downs = nn.ModuleList([])
        self.middle = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Build encoder
        for i in range(self.depth):
            if i == 0:
                dim_in = self.Ns[i]
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i - 1]
                dim_out = self.Ns[i]

            if i < (self.depth - 1):
                self.downs.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in, dim_out, self.use_norm, bias=False),
                        Downsample(self.Ss[i]),
                        CombinerDown("sum", 1, dim_out, bias=False)
                    ])
                )
            elif i == (self.depth - 1):
                self.downs.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in, dim_out, self.use_norm, bias=False),
                    ])
                )

        # Middle block
        self.middle.append(nn.ModuleList([
            ResnetBlock(self.Ns[self.depth], self.Ns[self.depth], self.use_norm, bias=False)
        ]))

        # Build decoder
        for i in range(self.depth - 1, -1, -1):
            if i == 0:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i]
            else:
                dim_in = self.Ns[i] * 2
                dim_out = self.Ns[i - 1]

            if i > 0:
                self.ups.append(nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, use_norm=self.use_norm, bias=False),
                    Upsample(self.Ss[i]),
                    CombinerUp("sum", 1, dim_out, bias=False)
                ]))
            elif i == 0:
                self.ups.append(
                    nn.ModuleList([
                        ResnetBlock(dim_in, dim_out, use_norm=self.use_norm, bias=False),
                    ])
                )

        self.cropconcat = CropConcatBlock()

    def setup_CQT_len(self, length):
        """Compatibility method - does nothing for time-domain model."""
        pass

    def forward(self, inputs, sigma):
        """
        Forward pass.

        Args:
            inputs: Input signal, shape (B, T)
            sigma: Noise levels, shape (B, 1)

        Returns:
            Predicted signal, shape (B, T)
        """
        # Embed noise level
        sigma = self.embedding(sigma)

        # Add channel dimension: (B, T) -> (B, 1, T)
        x = inputs.unsqueeze(1)
        pyr = x

        # Initial convolution
        x = self.init_conv(x)

        # Encoder
        hs = []
        for i, modules in enumerate(self.downs):
            if i < (self.depth - 1):
                resnet, downsample, combiner = modules
                x = resnet(x, sigma)
                hs.append(x)
                x = downsample(x)
                pyr = downsample(pyr)
                x = combiner(pyr, x)
            elif i == (self.depth - 1):
                (resnet,) = modules
                x = resnet(x, sigma)
                hs.append(x)

        # Middle
        for modules in self.middle:
            (resnet,) = modules
            x = resnet(x, sigma)

        # Decoder
        pyr = None
        for i, modules in enumerate(self.ups):
            j = self.depth - i - 1
            if j > 0:
                resnet, upsample, combiner = modules
                skip = hs.pop()
                x = self.cropconcat(x, skip)
                x = resnet(x, sigma)
                pyr = combiner(pyr, x)
                x = upsample(x)
                pyr = upsample(pyr)
            elif j == 0:
                (resnet,) = modules
                skip = hs.pop()
                x = self.cropconcat(x, skip)
                x = resnet(x, sigma)
                pyr = combiner(pyr, x)

        # Remove channel dimension: (B, 1, T) -> (B, T)
        pred = pyr.squeeze(1)

        # Ensure output has same shape as input
        assert pred.shape == inputs.shape, f"Shape mismatch: {pred.shape} vs {inputs.shape}"

        return pred
