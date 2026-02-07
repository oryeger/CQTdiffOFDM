"""
Conditional 1D U-Net for OFDM Declipping Diffusion Model.

Input conditioning:
- y: clipped time-domain IQ (2 channels: I, Q)
- m: clipping mask (1 channel, 1 where |y| hits clip threshold)
- A: optional clip level embedded as scalar

Output: noise prediction ε̂ for DDPM/DDIM

Architecture:
- 4-5 downsample stages (stride 2), channels 64→128→256→512(→512)
- 2 ResBlocks per stage with dilated convolutions (dilation cycle {1,2,4,8})
- Timestep embedding: sinusoidal t → MLP → FiLM (scale/shift) injection
- GroupNorm (no BatchNorm) for mixed FFT sizes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, List


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal embeddings for timestep conditioning."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps, shape (B,) or (B, 1)
        Returns:
            Embeddings, shape (B, dim)
        """
        if t.ndim == 2:
            t = t.squeeze(-1)
        
        device = t.device
        half_dim = self.dim // 2
        
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half_dim, device=device) / half_dim
        )
        args = t[:, None] * freqs[None, :]
        
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        
        return embedding


class TimestepMLPEmbedding(nn.Module):
    """MLP to process timestep embedding for FiLM conditioning."""
    
    def __init__(self, time_dim: int, embed_dim: int):
        super().__init__()
        self.sinusoidal = SinusoidalPositionEmbedding(time_dim)
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps, shape (B,) or (B, 1)
        Returns:
            Embedding, shape (B, embed_dim)
        """
        return self.mlp(self.sinusoidal(t))


class ClipLevelEmbedding(nn.Module):
    """Optional embedding for clip level A."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )
    
    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            A: Clip level, shape (B,) or (B, 1)
        Returns:
            Embedding, shape (B, embed_dim)
        """
        if A.ndim == 1:
            A = A.unsqueeze(-1)
        return self.mlp(A)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation for conditioning."""
    
    def __init__(self, embed_dim: int, out_channels: int):
        super().__init__()
        self.projection = nn.Linear(embed_dim, 2 * out_channels)
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features, shape (B, C, T)
            emb: Conditioning embedding, shape (B, embed_dim)
        Returns:
            Modulated features, shape (B, C, T)
        """
        params = self.projection(emb)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)
        return x * (1 + gamma) + beta


class DilatedResBlock(nn.Module):
    """
    Residual block with dilated 1D convolutions and FiLM conditioning.
    
    Uses:
    - Dilated conv (kernel=3) with configurable dilation
    - SiLU activation
    - GroupNorm (avoids BatchNorm issues with varying FFT sizes)
    - FiLM conditioning from timestep embedding
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int,
        kernel_size: int = 3,
        dilation: int = 1,
        num_groups: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv path
        padding = (kernel_size - 1) * dilation // 2
        
        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        # FiLM modulation
        self.film = FiLM(embed_dim, out_channels)
        
        # Second conv path
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        
        # Activation
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (B, C_in, T)
            emb: Timestep embedding, shape (B, embed_dim)
        Returns:
            Output features, shape (B, C_out, T)
        """
        h = x
        
        # First conv
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # FiLM conditioning
        h = self.film(h, emb)
        
        # Second conv
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Residual connection with scaling for stability
        return (h + self.skip(x)) / np.sqrt(2)


class ResBlockStack(nn.Module):
    """
    Stack of residual blocks with cycling dilations.
    
    Dilation cycle: {1, 2, 4, 8} for multi-scale receptive field.
    """
    
    def __init__(
        self,
        channels: int,
        embed_dim: int,
        num_blocks: int = 2,
        kernel_size: int = 3,
        dilation_cycle: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = dilation_cycle[i % len(dilation_cycle)]
            self.blocks.append(
                DilatedResBlock(
                    channels, channels, embed_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, emb)
        return x


class Downsample1D(nn.Module):
    """Downsample by factor of 2 using strided convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    """Upsample by factor of 2 using transposed convolution."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalUNet1D(nn.Module):
    """
    Conditional 1D U-Net for OFDM Declipping.
    
    Inputs:
    - x_t: Noisy estimate at timestep t, shape (B, 2, T) [I, Q channels]
    - y: Clipped observation, shape (B, 2, T)
    - m: Clipping mask, shape (B, 1, T) or (B, T)
    - t: Timestep, shape (B,) or (B, 1)
    - A: Optional clip level, shape (B,) or (B, 1)
    
    Output:
    - Predicted noise ε̂, shape (B, 2, T)
    
    Architecture:
    - Input: concatenate [x_t, y, m] → 5 channels → initial conv
    - Encoder: 4-5 stages with downsample
    - Middle: residual blocks
    - Decoder: 4-5 stages with upsample + skip connections
    - Output: conv to 2 channels (I, Q)
    """
    
    def __init__(
        self,
        in_channels: int = 2,  # I, Q
        out_channels: int = 2,  # Predicted noise for I, Q
        cond_channels: int = 3,  # y (2) + m (1)
        base_channels: int = 64,
        channel_mults: Tuple[int, ...] = (1, 2, 4, 8),  # 64, 128, 256, 512
        num_res_blocks: int = 2,
        embed_dim: int = 512,
        dilation_cycle: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.0,
        use_clip_level: bool = True,
        cond_drop_prob: float = 0.1,  # For classifier-free guidance
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.use_clip_level = use_clip_level
        self.cond_drop_prob = cond_drop_prob
        self.num_levels = len(channel_mults)
        
        # Total input channels: x_t + y + m
        total_in = in_channels + cond_channels
        
        # Timestep embedding
        time_dim = base_channels * 4
        self.time_embed = TimestepMLPEmbedding(time_dim, embed_dim)
        
        # Optional clip level embedding
        if use_clip_level:
            self.clip_level_embed = ClipLevelEmbedding(embed_dim)
        
        # Initial convolution
        self.init_conv = nn.Conv1d(total_in, base_channels, kernel_size=7, padding=3)
        
        # Null conditioning embedding for classifier-free guidance
        self.null_cond = nn.Parameter(torch.randn(1, cond_channels, 1) * 0.01)
        
        # ========== ENCODER ==========
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        
        channels = base_channels
        encoder_channels = [channels]
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            # ResBlocks at this level
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlockStack(
                        channels, embed_dim, num_blocks=1,
                        dilation_cycle=dilation_cycle, dropout=dropout
                    )
                )
                if channels != out_ch:
                    self.encoder_blocks.append(nn.Conv1d(channels, out_ch, 1))
                    channels = out_ch
                encoder_channels.append(channels)
            
            # Downsample (except at last level)
            if level < len(channel_mults) - 1:
                self.encoder_downsamples.append(Downsample1D(channels))
                encoder_channels.append(channels)
        
        # ========== MIDDLE ==========
        self.middle_block = nn.ModuleList([
            ResBlockStack(channels, embed_dim, num_blocks=2, dilation_cycle=dilation_cycle),
            ResBlockStack(channels, embed_dim, num_blocks=2, dilation_cycle=dilation_cycle),
        ])
        
        # ========== DECODER ==========
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            # ResBlocks with skip connections
            for i in range(num_res_blocks + 1):
                skip_ch = encoder_channels.pop()
                
                self.decoder_blocks.append(
                    nn.ModuleList([
                        nn.Conv1d(channels + skip_ch, out_ch, 1),  # Merge skip
                        ResBlockStack(
                            out_ch, embed_dim, num_blocks=1,
                            dilation_cycle=dilation_cycle, dropout=dropout
                        ),
                    ])
                )
                channels = out_ch
            
            # Upsample (except at first level)
            if level > 0:
                self.decoder_upsamples.append(Upsample1D(channels))
        
        # ========== OUTPUT ==========
        self.out_norm = nn.GroupNorm(8, channels)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(channels, out_channels, kernel_size=7, padding=3)
        
        # Initialize output with zeros for stable training
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        cond_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x_t: Noisy signal at timestep t, shape (B, 2, T)
            t: Timestep (noise level), shape (B,) or (B, 1)
            y: Clipped observation, shape (B, 2, T)
            m: Clipping mask (1 where clipped), shape (B, 1, T) or (B, T)
            A: Optional clip level, shape (B,) or (B, 1)
            cond_drop_mask: Optional mask for classifier-free guidance dropout
        
        Returns:
            Predicted noise ε̂, shape (B, 2, T)
        """
        B, _, T = x_t.shape
        
        # Handle mask shape
        if m.ndim == 2:
            m = m.unsqueeze(1)  # (B, T) -> (B, 1, T)
        
        # Classifier-free guidance: randomly drop conditioning
        if self.training and self.cond_drop_prob > 0:
            if cond_drop_mask is None:
                cond_drop_mask = torch.rand(B, device=x_t.device) < self.cond_drop_prob
            
            # Replace conditioning with learned null embedding
            null_cond = self.null_cond.expand(B, -1, T)
            y_m = torch.cat([y, m], dim=1)  # (B, 3, T)
            
            cond_drop_mask = cond_drop_mask.view(B, 1, 1).expand(-1, self.cond_channels, T)
            y_m = torch.where(cond_drop_mask, null_cond, y_m)
            
            y = y_m[:, :2, :]
            m = y_m[:, 2:3, :]
        
        # Timestep embedding
        emb = self.time_embed(t)
        
        # Optional clip level embedding
        if self.use_clip_level and A is not None:
            emb = emb + self.clip_level_embed(A)
        
        # Concatenate input with conditioning
        h = torch.cat([x_t, y, m], dim=1)  # (B, 5, T)
        
        # Initial conv
        h = self.init_conv(h)
        
        # ========== ENCODER ==========
        skips = [h]
        down_idx = 0
        
        for block in self.encoder_blocks:
            if isinstance(block, ResBlockStack):
                h = block(h, emb)
            else:
                h = block(h)  # 1x1 conv for channel change
            skips.append(h)
        
            # Check if we should downsample
            if down_idx < len(self.encoder_downsamples):
                # Downsample after all blocks at this level
                blocks_per_level = len(self.encoder_blocks) // self.num_levels
                if len(skips) % (blocks_per_level + 1) == 0:
                    h = self.encoder_downsamples[down_idx](h)
                    skips.append(h)
                    down_idx += 1
        
        # ========== MIDDLE ==========
        for block in self.middle_block:
            h = block(h, emb)
        
        # ========== DECODER ==========
        up_idx = 0
        
        for i, block in enumerate(self.decoder_blocks):
            # Get skip connection
            skip = skips.pop() if skips else None
            
            if skip is not None:
                # Handle size mismatch due to downsampling
                if h.shape[-1] != skip.shape[-1]:
                    diff = skip.shape[-1] - h.shape[-1]
                    h = F.pad(h, (diff // 2, diff - diff // 2))
                
                h = torch.cat([h, skip], dim=1)
            
            # Apply blocks
            merge_conv, res_block = block
            h = merge_conv(h)
            h = res_block(h, emb)
            
            # Upsample
            if up_idx < len(self.decoder_upsamples):
                blocks_per_level_dec = len(self.decoder_blocks) // self.num_levels
                if (i + 1) % (blocks_per_level_dec) == 0:
                    h = self.decoder_upsamples[up_idx](h)
                    up_idx += 1
        
        # ========== OUTPUT ==========
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        
        # Ensure output matches input size
        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode='linear', align_corners=False)
        
        return h
    
    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        
        eps_guided = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
        """
        if cfg_scale == 1.0:
            return self.forward(x_t, t, y, m, A, cond_drop_mask=None)
        
        B = x_t.shape[0]
        
        # Unconditional prediction
        cond_drop_mask = torch.ones(B, device=x_t.device, dtype=torch.bool)
        eps_uncond = self.forward(x_t, t, y, m, A, cond_drop_mask=cond_drop_mask)
        
        # Conditional prediction
        cond_drop_mask = torch.zeros(B, device=x_t.device, dtype=torch.bool)
        eps_cond = self.forward(x_t, t, y, m, A, cond_drop_mask=cond_drop_mask)
        
        # Classifier-free guidance
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


class ConditionalUNet1DSimple(nn.Module):
    """
    Simplified conditional 1D U-Net for OFDM declipping.
    
    More straightforward implementation with cleaner encoder-decoder structure.
    """
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        embed_dim: int = 512,
        dilation_cycle: Tuple[int, ...] = (1, 2, 4, 8),
        use_clip_level: bool = True,
        cond_drop_prob: float = 0.1,
    ):
        super().__init__()
        
        self.depth = depth
        self.use_clip_level = use_clip_level
        self.cond_drop_prob = cond_drop_prob
        
        # Conditioning: y (2 channels) + mask (1 channel) = 3 channels
        cond_channels = 3
        total_in = in_channels + cond_channels  # 5 channels
        
        # Channel progression: 64 -> 128 -> 256 -> 512
        self.channels = [base_channels * (2 ** min(i, 3)) for i in range(depth + 1)]
        
        # Time embedding
        time_dim = base_channels * 4
        self.time_embed = TimestepMLPEmbedding(time_dim, embed_dim)
        
        # Clip level embedding
        if use_clip_level:
            self.clip_level_embed = ClipLevelEmbedding(embed_dim)
        
        # Null conditioning for CFG
        self.null_cond = nn.Parameter(torch.randn(1, cond_channels, 1) * 0.01)
        
        # Initial conv
        self.init_conv = nn.Conv1d(total_in, self.channels[0], kernel_size=7, padding=3)
        
        # Encoder
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        for i in range(depth):
            self.down_blocks.append(
                nn.ModuleList([
                    DilatedResBlock(
                        self.channels[i], self.channels[i], embed_dim,
                        dilation=dilation_cycle[0]
                    ),
                    DilatedResBlock(
                        self.channels[i], self.channels[i], embed_dim,
                        dilation=dilation_cycle[1 % len(dilation_cycle)]
                    ),
                ])
            )
            self.down_samples.append(
                nn.Sequential(
                    nn.Conv1d(self.channels[i], self.channels[i + 1], 1),
                    nn.Conv1d(self.channels[i + 1], self.channels[i + 1], 4, stride=2, padding=1),
                )
            )
        
        # Middle
        self.middle = nn.ModuleList([
            DilatedResBlock(self.channels[-1], self.channels[-1], embed_dim, dilation=1),
            DilatedResBlock(self.channels[-1], self.channels[-1], embed_dim, dilation=2),
            DilatedResBlock(self.channels[-1], self.channels[-1], embed_dim, dilation=4),
            DilatedResBlock(self.channels[-1], self.channels[-1], embed_dim, dilation=8),
        ])
        
        # Decoder
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        for i in range(depth - 1, -1, -1):
            self.up_samples.append(
                nn.ConvTranspose1d(self.channels[i + 1], self.channels[i], 4, stride=2, padding=1)
            )
            # Input is upsampled + skip
            self.up_blocks.append(
                nn.ModuleList([
                    nn.Conv1d(self.channels[i] * 2, self.channels[i], 1),
                    DilatedResBlock(
                        self.channels[i], self.channels[i], embed_dim,
                        dilation=dilation_cycle[0]
                    ),
                    DilatedResBlock(
                        self.channels[i], self.channels[i], embed_dim,
                        dilation=dilation_cycle[1 % len(dilation_cycle)]
                    ),
                ])
            )
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, self.channels[0]),
            nn.SiLU(),
            nn.Conv1d(self.channels[0], out_channels, kernel_size=7, padding=3),
        )
        
        # Zero init output
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        cond_drop_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x_t: Noisy signal, shape (B, 2, T)
            t: Timestep, shape (B,) or (B, 1)
            y: Clipped observation, shape (B, 2, T)
            m: Clipping mask, shape (B, 1, T) or (B, T)
            A: Optional clip level, shape (B,) or (B, 1)
            cond_drop_mask: Optional bool mask for CFG training
        
        Returns:
            Predicted noise, shape (B, 2, T)
        """
        B, _, T = x_t.shape
        
        # Handle mask shape
        if m.ndim == 2:
            m = m.unsqueeze(1)
        
        # CFG dropout during training
        if self.training and self.cond_drop_prob > 0:
            if cond_drop_mask is None:
                cond_drop_mask = torch.rand(B, device=x_t.device) < self.cond_drop_prob
            
            null_cond = self.null_cond.expand(B, -1, T)
            y_m = torch.cat([y, m], dim=1)
            
            cond_drop_mask_expanded = cond_drop_mask.view(B, 1, 1).expand(-1, 3, T)
            y_m = torch.where(cond_drop_mask_expanded, null_cond, y_m)
            
            y = y_m[:, :2, :]
            m = y_m[:, 2:3, :]
        
        # Time embedding
        emb = self.time_embed(t)
        
        # Clip level embedding
        if self.use_clip_level and A is not None:
            emb = emb + self.clip_level_embed(A)
        
        # Concatenate conditioning
        h = torch.cat([x_t, y, m], dim=1)
        h = self.init_conv(h)
        
        # Encoder
        skips = []
        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            for block in down_block:
                h = block(h, emb)
            skips.append(h)
            h = down_sample(h)
        
        # Middle
        for block in self.middle:
            h = block(h, emb)
        
        # Decoder
        for up_sample, up_block, skip in zip(
            self.up_samples, self.up_blocks, reversed(skips)
        ):
            h = up_sample(h)
            
            # Handle size mismatch
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, diff))
            
            h = torch.cat([h, skip], dim=1)
            
            merge_conv = up_block[0]
            res_blocks = up_block[1:]
            
            h = merge_conv(h)
            for block in res_blocks:
                h = block(h, emb)
        
        # Output
        out = self.out_conv(h)
        
        # Ensure size matches
        if out.shape[-1] != T:
            out = F.interpolate(out, size=T, mode='linear', align_corners=False)
        
        return out
    
    def forward_with_cfg(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        if cfg_scale == 1.0:
            return self.forward(x_t, t, y, m, A)
        
        B = x_t.shape[0]
        
        # Unconditional
        eps_uncond = self.forward(
            x_t, t, y, m, A,
            cond_drop_mask=torch.ones(B, device=x_t.device, dtype=torch.bool)
        )
        
        # Conditional
        eps_cond = self.forward(
            x_t, t, y, m, A,
            cond_drop_mask=torch.zeros(B, device=x_t.device, dtype=torch.bool)
        )
        
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# For testing
if __name__ == "__main__":
    # Test model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ConditionalUNet1DSimple(
        in_channels=2,
        out_channels=2,
        base_channels=64,
        depth=4,
        embed_dim=512,
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model) / 1e6:.2f}M")
    
    # Test forward pass
    B, T = 4, 8192
    x_t = torch.randn(B, 2, T).to(device)
    t = torch.rand(B).to(device)
    y = torch.randn(B, 2, T).to(device)
    m = torch.randint(0, 2, (B, 1, T)).float().to(device)
    A = torch.rand(B).to(device)
    
    with torch.no_grad():
        out = model(x_t, t, y, m, A)
    
    print(f"Input shape: {x_t.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x_t.shape, "Output shape mismatch!"
    print("✓ Model test passed!")
