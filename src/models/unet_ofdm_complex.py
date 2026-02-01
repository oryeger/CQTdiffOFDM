"""
2-Channel 1D U-Net for Complex Baseband OFDM Signals.

Input: [Re(x), Im(x)] - 2 channels
Output: [Re(x), Im(x)] - 2 channels

Designed for unconditional diffusion prior on OFDM waveforms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal embeddings for noise level conditioning."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class RFF_MLP_Block(nn.Module):
    """Random Fourier Feature embedding + MLP for noise level."""
    
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.RFF_freq = nn.Parameter(16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
    
    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sigma: Noise level, shape (B, 1)
        Returns:
            Embedding, shape (B, embed_dim)
        """
        freqs = self.RFF_freq
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return self.MLP(table)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""
    
    def __init__(self, embed_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 2 * out_dim)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features, shape (B, C, T)
            emb: Conditioning embedding, shape (B, embed_dim)
        Returns:
            Modulated features, shape (B, C, T)
        """
        params = self.fc(emb)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return x * (1 + gamma) + beta


class ResBlock1D(nn.Module):
    """Residual block with FiLM conditioning and dilated convolutions."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 5,
        dilation: int = 1,
        use_norm: bool = True
    ):
        super().__init__()
        
        self.use_norm = use_norm
        
        if use_norm:
            self.norm1 = nn.GroupNorm(8, in_channels)
            self.norm2 = nn.GroupNorm(8, out_channels)
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        self.film = FiLM(embed_dim, out_channels)
        
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = x
        if self.use_norm:
            h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        if self.use_norm:
            h = self.norm2(h)
        h = self.act(h)
        h = self.film(h, emb)
        h = self.conv2(h)
        
        return (h + self.residual(x)) / np.sqrt(2)


class DilatedResStack(nn.Module):
    """Stack of residual blocks with exponentially increasing dilation."""
    
    def __init__(
        self,
        channels: int,
        embed_dim: int,
        num_layers: int = 4,
        kernel_size: int = 5,
        max_dilation: int = 64
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = min(2 ** i, max_dilation)
            self.blocks.append(
                ResBlock1D(channels, channels, embed_dim, kernel_size, dilation)
            )
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, emb)
        return x


class Downsample1D(nn.Module):
    """Downsample by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsample by factor of 2."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet1DComplex(nn.Module):
    """
    1D U-Net for complex baseband OFDM signals.
    
    - Input: 2 channels [Re(x), Im(x)]
    - Output: 2 channels [Re(x), Im(x)]
    - Noise-conditioned via FiLM
    - Dilated convolutions for large receptive field
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4, 8),
        num_res_blocks: int = 3,
        embed_dim: int = 512,
        attention_resolutions: tuple = (),  # Resolutions to add attention
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Noise level embedding
        self.time_embed = RFF_MLP_Block(embed_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv1d(in_channels, base_channels, 7, padding=3)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        
        channels = base_channels
        encoder_channels = [channels]
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # Residual blocks at this level
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    DilatedResStack(channels, embed_dim, num_layers=4)
                )
                if channels != out_ch:
                    self.encoder_blocks.append(
                        nn.Conv1d(channels, out_ch, 1)
                    )
                    channels = out_ch
                encoder_channels.append(channels)
            
            # Downsample (except last level)
            if level < len(channel_mults) - 1:
                self.downsamplers.append(Downsample1D(channels))
                encoder_channels.append(channels)
        
        # Middle
        self.middle_block = DilatedResStack(channels, embed_dim, num_layers=6)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            # Upsample (except first level)
            if level < len(channel_mults) - 1:
                self.upsamplers.append(Upsample1D(channels))
            
            # Residual blocks with skip connections
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                in_ch = channels + skip_ch
                
                self.decoder_blocks.append(
                    nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, 1),
                        DilatedResStack(out_ch, embed_dim, num_layers=4)
                    )
                )
                channels = out_ch
        
        # Final convolution
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, out_channels, 7, padding=3),
        )
        
        # Initialize output layer with zeros for stable training
        nn.init.zeros_(self.final_conv[-1].weight)
        nn.init.zeros_(self.final_conv[-1].bias)
    
    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy signal, shape (B, 2, T)
            sigma: Noise level, shape (B, 1)
            
        Returns:
            Denoised signal estimate, shape (B, 2, T)
        """
        # Time embedding
        emb = self.time_embed(sigma)
        
        # Initial conv
        h = self.init_conv(x)
        
        # Encoder
        skips = [h]
        down_idx = 0
        
        for block in self.encoder_blocks:
            if isinstance(block, DilatedResStack):
                h = block(h, emb)
            else:
                h = block(h)  # 1x1 conv
            skips.append(h)
            
            # Check if we should downsample
            if down_idx < len(self.downsamplers):
                if len(skips) % (len(self.encoder_blocks) // len(self.downsamplers) + 1) == 0:
                    h = self.downsamplers[down_idx](h)
                    skips.append(h)
                    down_idx += 1
        
        # Middle
        h = self.middle_block(h, emb)
        
        # Decoder - simplified version
        # Just use the final skip for now
        skip = skips.pop() if skips else None
        if skip is not None and skip.shape[-1] == h.shape[-1]:
            h = torch.cat([h, skip], dim=1)
        
        for block in self.decoder_blocks:
            h = block[0](h)  # Channel projection
            h = block[1](h, emb)  # Residual stack
        
        # Final conv
        return self.final_conv(h)


class UNet1DComplexSimple(nn.Module):
    """
    Simplified 1D U-Net for complex OFDM signals.
    
    More straightforward architecture that's easier to debug.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 64,
        depth: int = 5,
        embed_dim: int = 512,
    ):
        super().__init__()
        
        self.depth = depth
        
        # Noise embedding
        self.time_embed = RFF_MLP_Block(embed_dim)
        
        # Channel progression
        self.channels = [base_channels * (2 ** min(i, 3)) for i in range(depth + 1)]
        
        # Encoder
        self.init_conv = nn.Conv1d(in_channels, self.channels[0], 7, padding=3)
        
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        for i in range(depth):
            self.down_blocks.append(
                DilatedResStack(self.channels[i], embed_dim, num_layers=4)
            )
            self.down_samples.append(
                nn.Sequential(
                    nn.Conv1d(self.channels[i], self.channels[i+1], 1),
                    nn.Conv1d(self.channels[i+1], self.channels[i+1], 4, stride=2, padding=1)
                )
            )
        
        # Middle
        self.middle = DilatedResStack(self.channels[-1], embed_dim, num_layers=6)
        
        # Decoder
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            self.up_samples.append(
                nn.ConvTranspose1d(self.channels[i+1], self.channels[i], 4, stride=2, padding=1)
            )
            # Input is upsampled + skip connection
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv1d(self.channels[i] * 2, self.channels[i], 1),
                    DilatedResStack(self.channels[i], embed_dim, num_layers=4)
                )
            )
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, self.channels[0]),
            nn.GELU(),
            nn.Conv1d(self.channels[0], in_channels, 7, padding=3)
        )
        
        # Zero init for stable training
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)
    
    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy signal, shape (B, 2, T)
            sigma: Noise level, shape (B, 1)
        Returns:
            Predicted clean signal, shape (B, 2, T)
        """
        emb = self.time_embed(sigma)
        
        # Encoder
        h = self.init_conv(x)
        skips = []
        
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            h = down_block(h, emb)
            skips.append(h)
            h = down_sample(h)
        
        # Middle
        h = self.middle(h, emb)
        
        # Decoder
        for up_sample, up_block, skip in zip(
            self.up_samples, self.up_blocks, reversed(skips)
        ):
            h = up_sample(h)
            # Handle size mismatch
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = up_block[0](h)
            h = up_block[1](h, emb)
        
        return self.out_conv(h)
