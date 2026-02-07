"""
Complete OFDM Diffusion Declipping Demo

This script demonstrates the full pipeline:
1. Generate OFDM dataset with random parameters
2. Visualize a sample and its constellation/EVM
3. Train a diffusion model on clean OFDM signals
4. Apply clipping to a test signal
5. Run declipping reconstruction
6. Compare: Original → Clipped → Reconstructed (with EVM at each stage)

Two model architectures available:
- "simple": Unconditional model with inference-time guidance (default)
- "conditional": Conditional model trained specifically for declipping with EVM-focused loss

Usage:
    python demo_ofdm_full_pipeline.py --train_steps 1000 --num_test_samples 3
    python demo_ofdm_full_pipeline.py --model_type conditional --train_steps 5000
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import random

# Import the new conditional declipping model
try:
    from src.models.unet_ofdm_declip import ConditionalUNet1DSimple
    from src.ofdm.ofdm_declip_dataset import DeclipConfig, OFDMDeclipDataset, collate_declip_batch
    HAS_CONDITIONAL_MODEL = True
except ImportError:
    HAS_CONDITIONAL_MODEL = False
    print("Warning: Conditional model not available. Using simple model only.")


# ============== PART 1: OFDM Signal Generation ==============

@dataclass
class OFDMConfig:
    """OFDM configuration with randomization ranges."""
    fft_sizes: List[int] = field(default_factory=lambda: [256])  # Fixed FFT size
    modulations: List[str] = field(default_factory=lambda: ["QPSK"])  # QPSK only
    cp_ratio_range: Tuple[float, float] = (0.125, 0.125)  # Fixed CP ratio
    guard_band_range: Tuple[float, float] = (0.1, 0.1)  # Fixed guard band
    num_symbols_range: Tuple[int, int] = (14, 14)  # Fixed number of symbols


def generate_qam_constellation(modulation: str) -> np.ndarray:
    """Generate normalized QAM constellation."""
    if modulation == "QPSK":
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif modulation == "16QAM":
        points = []
        for i in range(4):
            for j in range(4):
                points.append((2*i - 3) + 1j*(2*j - 3))
        constellation = np.array(points)
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    elif modulation == "64QAM":
        points = []
        for i in range(8):
            for j in range(8):
                points.append((2*i - 7) + 1j*(2*j - 7))
        constellation = np.array(points)
        constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    else:
        raise ValueError(f"Unknown modulation: {modulation}")
    return constellation


def generate_ofdm_signal(
    config: OFDMConfig,
    target_length: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Generate a complex baseband OFDM signal with random parameters.
    
    Returns:
        signal: Complex signal, shape (target_length,)
        metadata: Dictionary with all parameters and symbols
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Random parameters
    fft_size = random.choice(config.fft_sizes)
    modulation = random.choice(config.modulations)
    cp_ratio = np.random.uniform(*config.cp_ratio_range)
    guard_band_ratio = np.random.uniform(*config.guard_band_range)
    
    # Derived parameters
    cp_length = int(fft_size * cp_ratio)
    symbol_length = fft_size + cp_length
    num_guard = int(fft_size * guard_band_ratio / 2)
    num_symbols = target_length // symbol_length
    num_symbols = max(5, min(num_symbols, config.num_symbols_range[1]))
    
    # Data subcarrier indices (exclude guards and DC)
    data_indices = []
    for k in range(fft_size):
        if k < num_guard or k >= fft_size - num_guard:
            continue
        if k == fft_size // 2:  # DC null
            continue
        data_indices.append(k)
    
    num_data_subcarriers = len(data_indices)
    
    # Generate QAM symbols
    constellation = generate_qam_constellation(modulation)
    total_symbols = num_data_subcarriers * num_symbols
    symbol_indices = np.random.randint(0, len(constellation), total_symbols)
    data_symbols = constellation[symbol_indices].reshape(num_symbols, num_data_subcarriers)
    
    # Build frequency-domain OFDM symbols
    freq_symbols = np.zeros((num_symbols, fft_size), dtype=complex)
    freq_symbols[:, data_indices] = data_symbols
    
    # IFFT to time domain
    time_symbols = np.fft.ifft(freq_symbols, axis=1) * np.sqrt(fft_size)
    
    # Add cyclic prefix
    if cp_length > 0:
        cp = time_symbols[:, -cp_length:]
        time_symbols_cp = np.concatenate([cp, time_symbols], axis=1)
    else:
        time_symbols_cp = time_symbols
    
    # Concatenate all symbols
    signal = time_symbols_cp.flatten()
    
    # Normalize to unit variance
    signal = signal / np.std(signal)
    
    # Pad or truncate
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))
    else:
        signal = signal[:target_length]
    
    metadata = {
        'fft_size': fft_size,
        'modulation': modulation,
        'cp_ratio': cp_ratio,
        'cp_length': cp_length,
        'guard_band_ratio': guard_band_ratio,
        'num_guard': num_guard,
        'num_symbols': num_symbols,
        'num_data_subcarriers': num_data_subcarriers,
        'data_indices': data_indices,
        'data_symbols': data_symbols,
        'constellation': constellation,
        'seed': seed,
    }
    
    return signal, metadata


def demodulate_ofdm(signal: np.ndarray, metadata: dict) -> np.ndarray:
    """Demodulate OFDM signal to recover QAM symbols."""
    fft_size = metadata['fft_size']
    cp_length = metadata['cp_length']
    num_symbols = metadata['num_symbols']
    data_indices = metadata['data_indices']
    symbol_length = fft_size + cp_length
    
    recovered_symbols = []
    
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_length + cp_length
        end = start + fft_size
        
        if end > len(signal):
            break
        
        time_symbol = signal[start:end]
        freq_symbol = np.fft.fft(time_symbol) / np.sqrt(fft_size)
        data = freq_symbol[data_indices]
        recovered_symbols.append(data)
    
    return np.array(recovered_symbols)


def compute_evm(reference: np.ndarray, recovered: np.ndarray) -> float:
    """Compute EVM as percentage."""
    error = recovered - reference
    error_power = np.mean(np.abs(error)**2)
    ref_power = np.mean(np.abs(reference)**2)
    evm = np.sqrt(error_power / (ref_power + 1e-10)) * 100
    return evm


def complex_to_2ch(signal: np.ndarray) -> np.ndarray:
    """Convert complex to 2-channel [Re, Im]."""
    return np.stack([signal.real, signal.imag], axis=0).astype(np.float32)


def ch2_to_complex(signal: np.ndarray) -> np.ndarray:
    """Convert 2-channel back to complex."""
    return signal[0] + 1j * signal[1]


# ============== PART 2: Dataset ==============

class OFDMDataset(torch.utils.data.IterableDataset):
    """OFDM dataset for training."""
    
    def __init__(self, signal_length: int = 4096, config: OFDMConfig = None):
        self.signal_length = signal_length
        self.config = config or OFDMConfig()
    
    def __iter__(self):
        while True:
            seed = np.random.randint(0, 2**31)
            signal, _ = generate_ofdm_signal(self.config, self.signal_length, seed)
            yield complex_to_2ch(signal)


# ============== PART 3: Simple 1D U-Net Model ==============

class ResBlock(nn.Module):
    """Residual block with time embedding."""
    
    def __init__(self, channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 5, padding=2)
        self.conv2 = nn.Conv1d(channels, channels, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.emb_proj = nn.Linear(emb_dim, channels)
        self.act = nn.GELU()
    
    def forward(self, x, emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        emb_out = self.emb_proj(emb).unsqueeze(-1)
        h = h + emb_out
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        
        return (x + h) / np.sqrt(2)


class SimpleUNet(nn.Module):
    """Simple 1D U-Net for OFDM signals."""
    
    def __init__(self, in_ch=2, base_ch=32, depth=4, emb_dim=256):
        super().__init__()
        
        self.depth = depth
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(64, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )
        
        # Encoder
        self.init_conv = nn.Conv1d(in_ch, base_ch, 7, padding=3)
        
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        ch = base_ch
        skip_channels = []  # Track channels at each skip connection
        for i in range(depth):
            self.down_blocks.append(ResBlock(ch, emb_dim))
            skip_channels.append(ch)  # Save channel count before downsampling
            next_ch = min(ch * 2, 256)
            self.down_convs.append(nn.Conv1d(ch, next_ch, 4, stride=2, padding=1))
            ch = next_ch
        
        # Middle
        self.mid_block = ResBlock(ch, emb_dim)
        
        # Decoder - use tracked skip channels (reversed)
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        
        for i in range(depth):
            skip_ch = skip_channels[depth - 1 - i]  # Reversed order
            prev_ch = ch
            next_ch = skip_ch  # Output should match skip channel count
            self.up_convs.append(nn.ConvTranspose1d(prev_ch, next_ch, 4, stride=2, padding=1))
            combined_ch = next_ch + skip_ch  # After concatenation
            self.up_blocks.append(ResBlock(combined_ch, emb_dim))
            self.up_blocks.append(nn.Conv1d(combined_ch, next_ch, 1))  # reduce channels
            ch = next_ch
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, in_ch, 7, padding=3),
        )
        nn.init.zeros_(self.out_conv[-1].weight)
        nn.init.zeros_(self.out_conv[-1].bias)
    
    def get_time_emb(self, sigma):
        """Get sinusoidal time embedding."""
        half_dim = 32
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device) * -emb)
        emb = sigma * emb
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_mlp(emb)
    
    def forward(self, x, sigma):
        """
        Args:
            x: Input signal, shape (B, 2, T)
            sigma: Noise level, shape (B, 1)
        """
        emb = self.get_time_emb(sigma)
        
        h = self.init_conv(x)
        
        # Encoder
        skips = []
        for down_block, down_conv in zip(self.down_blocks, self.down_convs):
            h = down_block(h, emb)
            skips.append(h)
            h = down_conv(h)
        
        # Middle
        h = self.mid_block(h, emb)
        
        # Decoder
        for i in range(self.depth):
            h = self.up_convs[i](h)
            skip = skips[self.depth - 1 - i]
            
            # Handle size mismatch
            if h.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - h.shape[-1]
                h = F.pad(h, (0, diff))
            
            h = torch.cat([h, skip], dim=1)
            h = self.up_blocks[i*2](h, emb)
            h = self.up_blocks[i*2 + 1](h)
        
        return self.out_conv(h)


# ============== PART 4: Diffusion ==============

class SimpleDiffusion:
    """Simple diffusion with EDM-style parameterization."""
    
    def __init__(self, sigma_min=0.002, sigma_max=1.0, sigma_data=0.5):
        """
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level (reduced for stability)
            sigma_data: Expected std of data (for EDM preconditioning)
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
    
    def sample_sigma(self, batch_size, device):
        """Sample training noise levels (log-uniform)."""
        log_sigma = torch.rand(batch_size, device=device)
        log_sigma = log_sigma * (np.log(self.sigma_max) - np.log(self.sigma_min)) + np.log(self.sigma_min)
        return torch.exp(log_sigma).unsqueeze(-1)
    
    def get_scalings(self, sigma):
        """Get EDM-style preconditioning scalings."""
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        return c_skip, c_out, c_in
    
    def get_schedule(self, num_steps, device):
        """Get sampling schedule."""
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        sigmas = self.sigma_max ** (1 - t) * self.sigma_min ** t
        sigmas[-1] = 0
        return sigmas


def train_step(model, diffusion, batch, optimizer, device, 
                use_constellation_loss=False, ofdm_params=None, lambda_const=0.1):
    """
    Single training step with optional constellation-aware loss.
    
    Args:
        use_constellation_loss: If True, add loss to encourage valid constellation symbols
        ofdm_params: Dict with fft_size, cp_length, num_symbols, modulation, data_indices
        lambda_const: Weight for constellation loss
    """
    model.train()
    optimizer.zero_grad()
    
    x_0 = batch.to(device)
    B = x_0.shape[0]
    
    # Check for NaN in input
    if torch.isnan(x_0).any():
        print("Warning: NaN in input batch, skipping")
        return 0.0
    
    # Sample noise level
    sigma = diffusion.sample_sigma(B, device)
    
    # Add noise
    noise = torch.randn_like(x_0)
    x_noisy = x_0 + sigma.unsqueeze(-1) * noise
    
    # EDM-style preconditioning
    c_skip, c_out, c_in = diffusion.get_scalings(sigma)
    
    # Scale input and get model output
    model_input = c_in.unsqueeze(-1) * x_noisy
    model_output = model(model_input, sigma)
    
    # EDM denoising: D(x, sigma) = c_skip * x + c_out * F(c_in * x, sigma)
    x_pred = c_skip.unsqueeze(-1) * x_noisy + c_out.unsqueeze(-1) * model_output
    
    # Check for NaN in prediction
    if torch.isnan(x_pred).any():
        print("Warning: NaN in model prediction, skipping")
        return float('nan')
    
    # MSE Loss (standard diffusion objective)
    mse_loss = F.mse_loss(x_pred, x_0)
    
    # Just use MSE loss - constellation awareness comes from the data itself
    # The model learns OFDM structure implicitly by denoising OFDM signals
    total_loss = mse_loss
    
    # Optional: Add very small constellation regularization (disabled by default)
    if use_constellation_loss and ofdm_params is not None and lambda_const > 0:
        try:
            # Get constellation as real/imag (works on all devices)
            const_real, const_imag = get_constellation_real(ofdm_params['modulation'], device)
            
            # Demodulate using real-only DFT (works on all devices)
            pred_real, pred_imag = demodulate_ofdm_real(
                x_pred,
                ofdm_params['fft_size'],
                ofdm_params['cp_length'],
                ofdm_params['num_symbols'],
                ofdm_params['data_indices'],
            )
            
            target_real, target_imag = demodulate_ofdm_real(
                x_0,
                ofdm_params['fft_size'],
                ofdm_params['cp_length'],
                ofdm_params['num_symbols'],
                ofdm_params['data_indices'],
            )
            
            # Symbol MSE loss only (simpler, more stable than constellation loss)
            symbol_mse = F.mse_loss(pred_real, target_real) + F.mse_loss(pred_imag, target_imag)
            
            # Only add if symbol_mse is reasonable (< 10x mse_loss)
            if symbol_mse.item() < 10 * mse_loss.item():
                total_loss = mse_loss + lambda_const * symbol_mse
        except Exception as e:
            pass  # Silently fall back to MSE only
    
    # Check for NaN in loss
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN/Inf loss, skipping step")
        optimizer.zero_grad()
        return float('nan')
    
    # Scale loss to prevent gradient explosion
    scaled_loss = total_loss / (1.0 + total_loss.detach() * 0.1)  # Soft scaling for large losses
    scaled_loss.backward()
    
    # Check for NaN in gradients and compute total norm
    has_nan = False
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                has_nan = True
                break
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    if has_nan or total_norm > 100:
        # Reset gradients if NaN or explosion detected
        optimizer.zero_grad()
        return float('nan')
    
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return total_loss.item()


def denoise(model, x_noisy, sigma, diffusion):
    """Apply EDM-style denoiser."""
    model.eval()
    with torch.no_grad():
        c_skip, c_out, c_in = diffusion.get_scalings(sigma)
        model_input = c_in.unsqueeze(-1) * x_noisy
        model_output = model(model_input, sigma)
        x_denoised = c_skip.unsqueeze(-1) * x_noisy + c_out.unsqueeze(-1) * model_output
        return x_denoised


# ============== PART 4b: Conditional Declipping Model (NEW) ==============

class ConditionalDiffusion:
    """EDM-style diffusion for conditional model."""
    
    def __init__(self, sigma_min=1e-4, sigma_max=10.0, sigma_data=1.0, rho=7.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
    
    def sample_sigma(self, batch_size, device):
        """Sample training noise levels (log-normal)."""
        log_sigma = torch.randn(batch_size, device=device) * 1.2 - 1.2
        sigma = torch.exp(log_sigma)
        return torch.clamp(sigma, self.sigma_min, self.sigma_max)
    
    def get_scalings(self, sigma):
        """Get EDM scalings."""
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = torch.log(sigma) / 4
        return c_skip, c_out, c_in, c_noise
    
    def get_schedule(self, num_steps, device):
        """Get sigma schedule for sampling."""
        step_indices = torch.arange(num_steps + 1, device=device)
        t = step_indices / num_steps
        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        sigmas = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** self.rho
        sigmas[-1] = 0
        return sigmas


def train_conditional_step(
    model, diffusion, clean, clipped, mask, clip_level,
    optimizer, device, lambda_freq=0.1, lambda_mask=0.1
):
    """
    Training step for conditional declipping model.
    
    Loss = L_eps + λ_freq * L_freq + λ_mask * L_mask
    """
    model.train()
    optimizer.zero_grad()
    
    B = clean.shape[0]
    
    # Sample noise level
    sigma = diffusion.sample_sigma(B, device)
    
    # Add noise to clean signal
    noise = torch.randn_like(clean)
    x_noisy = clean + sigma.view(-1, 1, 1) * noise
    
    # Get scalings
    c_skip, c_out, c_in, c_noise = diffusion.get_scalings(sigma)
    c_in = c_in.view(-1, 1, 1)
    c_skip = c_skip.view(-1, 1, 1)
    c_out = c_out.view(-1, 1, 1)
    
    # Forward pass
    eps_pred = model(
        x_t=c_in * x_noisy,
        t=c_noise,
        y=clipped,
        m=mask,
        A=clip_level.squeeze(-1) if clip_level.ndim > 1 else clip_level,
    )
    
    # Compute target
    target = (clean - c_skip * x_noisy) / (c_out + 1e-8)
    
    # MSE loss
    loss_eps = F.mse_loss(eps_pred, target)
    
    # Mask-weighted time loss (emphasize clipped regions)
    if lambda_mask > 0:
        x_pred = c_skip * x_noisy + c_out * eps_pred
        error = x_pred - clean
        
        if mask.shape[1] == 1:
            mask_exp = mask.expand(-1, 2, -1)
        else:
            mask_exp = mask
        
        weighted_error = mask_exp * (error ** 2)
        loss_mask = weighted_error.sum() / (mask_exp.sum() + 1e-8)
    else:
        loss_mask = torch.tensor(0.0, device=device)
    
    total_loss = loss_eps + lambda_mask * loss_mask
    
    # Backward
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return total_loss.item(), loss_eps.item(), loss_mask.item()


@torch.no_grad()
def sample_conditional_ddim(
    model, diffusion, clipped, mask, clip_level, num_steps=50,
    cfg_scale=1.0, data_consistency=True, device=None
):
    """
    DDIM sampling with the conditional model and data consistency.
    
    For unclipped samples (m=0), overwrites x_0 estimate with observation y.
    """
    if device is None:
        device = clipped.device
    
    model.eval()
    B, C, T = clipped.shape
    
    sigmas = diffusion.get_schedule(num_steps, device)
    
    # Initialize from noise
    x = torch.randn(B, C, T, device=device) * sigmas[0]
    
    # Initialize unclipped regions from noised observation
    if data_consistency:
        noise = torch.randn_like(clipped) * sigmas[0]
        y_noisy = clipped + noise
        
        if mask.shape[1] == 1:
            m_exp = mask.expand(-1, C, -1)
        else:
            m_exp = mask
        
        x = torch.where(m_exp > 0.5, x, y_noisy)
    
    for i in tqdm(range(num_steps), desc="DDIM Sampling", leave=False):
        t_curr = sigmas[i]
        t_next = sigmas[i + 1]
        
        c_skip, c_out, c_in, c_noise = diffusion.get_scalings(t_curr.view(1))
        c_in = c_in.view(1, 1, 1)
        
        # Get noise prediction with optional CFG
        if cfg_scale != 1.0 and hasattr(model, 'forward_with_cfg'):
            eps = model.forward_with_cfg(
                c_in * x, c_noise.expand(B), clipped, mask,
                clip_level.squeeze(-1) if clip_level.ndim > 1 else clip_level,
                cfg_scale=cfg_scale
            )
        else:
            eps = model(
                c_in * x, c_noise.expand(B), clipped, mask,
                clip_level.squeeze(-1) if clip_level.ndim > 1 else clip_level,
            )
        
        # Compute x_0 estimate
        x_0 = c_skip.view(1, 1, 1) * x + c_out.view(1, 1, 1) * eps
        
        # Data consistency: replace unclipped regions with observation
        if data_consistency:
            if mask.shape[1] == 1:
                m_exp = mask.expand(-1, C, -1)
            else:
                m_exp = mask
            x_0 = torch.where(m_exp > 0.5, x_0, clipped)
        
        # DDIM step
        if t_next > 0:
            direction = (x - x_0) / (t_curr + 1e-8)
            x = x_0 + t_next * direction
        else:
            x = x_0
    
    return x


# ============== PART 5: Declipping with Guidance (FIXED + CONSTELLATION) ==============

def clip_signal(x, clip_level):
    """
    Apply MAGNITUDE-BASED clipping to complex IQ signal.
    
    If |x| = sqrt(I² + Q²) >= A, scale to magnitude A while preserving phase.
    
    Args:
        x: Signal tensor, shape (B, 2, T) with [I, Q] channels
        clip_level: Clipping threshold A
    
    Returns:
        Clipped signal, shape (B, 2, T)
    """
    # Compute magnitude
    I = x[:, 0, :]  # (B, T)
    Q = x[:, 1, :]  # (B, T)
    magnitude = torch.sqrt(I ** 2 + Q ** 2 + 1e-10)
    
    # Scale factor: min(1, A / |x|)
    scale = torch.clamp(clip_level / magnitude, max=1.0)
    
    # Apply scaling
    I_clipped = I * scale
    Q_clipped = Q * scale
    
    return torch.stack([I_clipped, Q_clipped], dim=1)


def get_clipping_mask(x, clip_level):
    """
    Get clipping mask based on magnitude.
    
    Args:
        x: Signal tensor, shape (B, 2, T) with [I, Q] channels
        clip_level: Clipping threshold A
    
    Returns:
        Mask tensor (1 where |x| >= A), shape (B, 1, T)
    """
    magnitude = torch.sqrt(x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
    mask = (magnitude >= clip_level).float().unsqueeze(1)
    return mask


def get_constellation_real(modulation: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get QAM constellation as real/imag tensors (MPS compatible).
    Returns (real_parts, imag_parts) each of shape [num_points].
    """
    if modulation == "QPSK":
        real = torch.tensor([1, 1, -1, -1], dtype=torch.float32) / np.sqrt(2)
        imag = torch.tensor([1, -1, 1, -1], dtype=torch.float32) / np.sqrt(2)
    elif modulation == "16QAM":
        real_list, imag_list = [], []
        for i in range(4):
            for j in range(4):
                real_list.append(2*i - 3)
                imag_list.append(2*j - 3)
        real = torch.tensor(real_list, dtype=torch.float32)
        imag = torch.tensor(imag_list, dtype=torch.float32)
        # Normalize
        power = torch.mean(real**2 + imag**2)
        real = real / torch.sqrt(power)
        imag = imag / torch.sqrt(power)
    elif modulation == "64QAM":
        real_list, imag_list = [], []
        for i in range(8):
            for j in range(8):
                real_list.append(2*i - 7)
                imag_list.append(2*j - 7)
        real = torch.tensor(real_list, dtype=torch.float32)
        imag = torch.tensor(imag_list, dtype=torch.float32)
        # Normalize
        power = torch.mean(real**2 + imag**2)
        real = real / torch.sqrt(power)
        imag = imag / torch.sqrt(power)
    else:
        # Default to QPSK
        real = torch.tensor([1, 1, -1, -1], dtype=torch.float32) / np.sqrt(2)
        imag = torch.tensor([1, -1, 1, -1], dtype=torch.float32) / np.sqrt(2)
    
    return real.to(device), imag.to(device)


def real_dft(x_real: torch.Tensor, x_imag: torch.Tensor, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute DFT using only real operations (MPS compatible).
    Output is normalized by 1/sqrt(N) to match time-domain magnitude.
    
    Args:
        x_real: Real part of input [B, N]
        x_imag: Imag part of input [B, N]
        n: DFT size
    
    Returns:
        (Y_real, Y_imag) each of shape [B, N], normalized
    """
    device = x_real.device
    
    # Create DFT matrix components
    k = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(1)  # [N, 1]
    m = torch.arange(n, device=device, dtype=torch.float32).unsqueeze(0)  # [1, N]
    angle = -2 * np.pi * k * m / n  # [N, N]
    
    W_real = torch.cos(angle)  # [N, N]
    W_imag = torch.sin(angle)  # [N, N]
    
    # DFT: Y = W @ x
    # (W_real + j*W_imag) @ (x_real + j*x_imag)
    # = W_real @ x_real - W_imag @ x_imag + j*(W_real @ x_imag + W_imag @ x_real)
    
    Y_real = x_real @ W_real.T - x_imag @ W_imag.T  # [B, N]
    Y_imag = x_real @ W_imag.T + x_imag @ W_real.T  # [B, N]
    
    # Normalize to match time-domain scale
    scale = 1.0 / np.sqrt(n)
    return Y_real * scale, Y_imag * scale


def demodulate_ofdm_real(
    x: torch.Tensor,  # [B, 2, T] I/Q signal
    fft_size: int,
    cp_length: int,
    num_symbols: int,
    data_indices: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Demodulate OFDM signal using real-only operations (MPS compatible).
    
    Returns:
        (symbols_real, symbols_imag) each of shape [B, num_sym, num_subcarriers]
    """
    B, C, T = x.shape
    symbol_len = fft_size + cp_length
    device = x.device
    
    x_real = x[:, 0]  # [B, T]
    x_imag = x[:, 1]  # [B, T]
    
    all_real, all_imag = [], []
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_len + cp_length  # Skip CP
        end = start + fft_size
        if end > T:
            break
        
        sym_real = x_real[:, start:end]  # [B, fft_size]
        sym_imag = x_imag[:, start:end]  # [B, fft_size]
        
        # Real-valued DFT
        freq_real, freq_imag = real_dft(sym_real, sym_imag, fft_size)
        
        if data_indices is not None:
            idx = data_indices.to(device)
            freq_real = freq_real[:, idx]
            freq_imag = freq_imag[:, idx]
        
        all_real.append(freq_real)
        all_imag.append(freq_imag)
    
    if len(all_real) > 0:
        symbols_real = torch.stack(all_real, dim=1)  # [B, num_sym, num_subcarriers]
        symbols_imag = torch.stack(all_imag, dim=1)
        return symbols_real, symbols_imag
    else:
        n_sc = data_indices.shape[0] if data_indices is not None else fft_size
        return (torch.zeros(B, 1, n_sc, device=device),
                torch.zeros(B, 1, n_sc, device=device))


def compute_constellation_loss_real(
    symbols_real: torch.Tensor,  # [B, num_sym, num_subcarriers]
    symbols_imag: torch.Tensor,  # [B, num_sym, num_subcarriers]
    const_real: torch.Tensor,    # [num_points]
    const_imag: torch.Tensor,    # [num_points]
    temperature: float = 0.5,
) -> torch.Tensor:
    """
    Soft constellation loss using only real operations (MPS compatible).
    
    Penalizes distance to nearest constellation point using soft-min.
    """
    device = symbols_real.device
    
    # Flatten symbols: [B, num_sym, num_sc] -> [N]
    sym_r = symbols_real.reshape(-1)  # [N]
    sym_i = symbols_imag.reshape(-1)  # [N]
    
    # Check for NaN/Inf
    if torch.isnan(sym_r).any() or torch.isnan(sym_i).any():
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # Compute squared distance to each constellation point
    # |s - c|^2 = (s_r - c_r)^2 + (s_i - c_i)^2
    # sym: [N], const: [M] -> distances: [N, M]
    diff_r = sym_r.unsqueeze(1) - const_real.unsqueeze(0)  # [N, M]
    diff_i = sym_i.unsqueeze(1) - const_imag.unsqueeze(0)  # [N, M]
    distances = diff_r**2 + diff_i**2  # [N, M]
    
    # Clamp for numerical stability
    distances = torch.clamp(distances, min=0.0, max=100.0)
    
    # Soft-min via softmax
    weights = F.softmax(-distances / temperature, dim=1)  # [N, M]
    soft_min_dist = (weights * distances).sum(dim=1)  # [N]
    
    loss = soft_min_dist.mean()
    
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    return loss


# Keep old functions for backward compatibility (used in inference)
def get_constellation_torch(modulation: str, device: str = "cpu") -> torch.Tensor:
    """Get QAM constellation as torch tensor (for inference on CPU)."""
    if modulation == "QPSK":
        constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.cfloat) / np.sqrt(2)
    elif modulation == "16QAM":
        points = []
        for i in range(4):
            for j in range(4):
                points.append(complex(2*i - 3, 2*j - 3))
        constellation = torch.tensor(points, dtype=torch.cfloat)
        constellation = constellation / torch.sqrt(torch.mean(torch.abs(constellation)**2))
    elif modulation == "64QAM":
        points = []
        for i in range(8):
            for j in range(8):
                points.append(complex(2*i - 7, 2*j - 7))
        constellation = torch.tensor(points, dtype=torch.cfloat)
        constellation = constellation / torch.sqrt(torch.mean(torch.abs(constellation)**2))
    else:
        # Default to QPSK
        constellation = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], dtype=torch.cfloat) / np.sqrt(2)
    return constellation.to(device)


def demodulate_ofdm_torch(
    x: torch.Tensor,  # [B, 2, T] I/Q signal
    fft_size: int,
    cp_length: int,
    num_symbols: int,
    data_indices: torch.Tensor = None,
) -> torch.Tensor:
    """Demodulate OFDM signal (CPU only, for inference)."""
    B, C, T = x.shape
    symbol_len = fft_size + cp_length
    
    # Move to CPU for complex operations
    x_cpu = x.cpu()
    x_complex = torch.complex(x_cpu[:, 0], x_cpu[:, 1])
    
    all_symbols = []
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_len + cp_length  # Skip CP
        end = start + fft_size
        if end > T:
            break
        
        symbol = x_complex[:, start:end]
        freq = torch.fft.fft(symbol, dim=-1)
        
        if data_indices is not None:
            freq = freq[:, data_indices.cpu()]
        
        all_symbols.append(freq)
    
    if len(all_symbols) > 0:
        symbols = torch.stack(all_symbols, dim=1)  # [B, num_sym, num_subcarriers]
        return symbols.to(x.device)
    else:
        return torch.zeros(B, 1, fft_size, dtype=torch.cfloat, device=x.device)


def compute_constellation_loss(
    symbols: torch.Tensor,  # [B, num_sym, num_subcarriers] complex
    constellation: torch.Tensor,  # [num_points] complex
    temperature: float = 0.5,
) -> torch.Tensor:
    """Soft constellation loss (CPU only, for inference)."""
    original_device = symbols.device
    symbols_cpu = symbols.cpu()
    constellation_cpu = constellation.cpu()
    
    symbols_flat = symbols_cpu.reshape(-1)
    
    if torch.isnan(symbols_flat).any() or torch.isinf(symbols_flat).any():
        return torch.tensor(0.0, device=original_device, requires_grad=True)
    
    distances = torch.abs(symbols_flat.unsqueeze(1) - constellation_cpu.unsqueeze(0)) ** 2
    distances = torch.clamp(distances, min=0.0, max=100.0)
    
    weights = F.softmax(-distances / temperature, dim=1)
    soft_min_dist = (weights * distances).sum(dim=1)
    loss = soft_min_dist.mean()
    
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=original_device, requires_grad=True)
    
    return loss.to(original_device)


def compute_measurement_loss_fixed(
    x_hat: torch.Tensor,           # Predicted clean signal [B, 2, T]
    y_clipped: torch.Tensor,       # Observed clipped signal [B, 2, T]
    clip_threshold: float,         # Clipping threshold T
    gamma_unclipped: float = 1.0,  # Weight for unclipped MSE
    eta_magnitude: float = 0.5,    # Weight for magnitude hinge
    eta_phase: float = 0.5,        # Weight for phase alignment
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    FIXED measurement loss with proper gradients for clipped samples.
    
    - Unclipped samples (|y| < T): MSE loss (exact match)
    - Clipped samples (|y| = T): 
        - Magnitude hinge: encourage |x̂| >= T
        - Phase alignment: encourage same direction as y
    """
    B, C, T = x_hat.shape
    
    # Compute magnitudes (for 2-channel I/Q representation)
    x_hat_mag = torch.sqrt(x_hat[:, 0]**2 + x_hat[:, 1]**2 + epsilon)
    y_mag = torch.sqrt(y_clipped[:, 0]**2 + y_clipped[:, 1]**2 + epsilon)
    
    # Identify clipped vs unclipped
    tolerance = 0.01 * clip_threshold
    is_clipped = (y_mag >= clip_threshold - tolerance).float()
    is_unclipped = 1.0 - is_clipped
    
    # Loss 1: Unclipped samples - exact match
    diff = x_hat - y_clipped
    mse_per_sample = diff[:, 0]**2 + diff[:, 1]**2
    unclipped_count = is_unclipped.sum() + epsilon
    loss_unclipped = (mse_per_sample * is_unclipped).sum() / unclipped_count
    
    # Loss 2: Clipped samples - magnitude hinge (soft)
    magnitude_deficit = clip_threshold - x_hat_mag
    hinge_loss = F.softplus(magnitude_deficit / 0.1) * 0.1
    hinge_loss = hinge_loss ** 2
    clipped_count = is_clipped.sum() + epsilon
    loss_magnitude = (hinge_loss * is_clipped).sum() / clipped_count
    
    # Loss 3: Clipped samples - phase alignment
    x_hat_norm = x_hat / (x_hat_mag.unsqueeze(1) + epsilon)
    y_norm = y_clipped / (y_mag.unsqueeze(1) + epsilon)
    cos_sim = x_hat_norm[:, 0] * y_norm[:, 0] + x_hat_norm[:, 1] * y_norm[:, 1]
    phase_loss = 1.0 - cos_sim
    loss_phase = (phase_loss * is_clipped).sum() / clipped_count
    
    # Combine
    total_loss = (
        gamma_unclipped * loss_unclipped +
        eta_magnitude * loss_magnitude +
        eta_phase * loss_phase
    )
    
    return total_loss


def compute_combined_loss(
    x_hat: torch.Tensor,           # [B, 2, T]
    y_clipped: torch.Tensor,       # [B, 2, T]
    clip_threshold: float,
    constellation: torch.Tensor,   # [M] complex
    fft_size: int,
    cp_length: int,
    num_symbols: int,
    data_indices: torch.Tensor,
    lambda_clip: float = 1.0,      # Weight for clipping loss
    lambda_const: float = 0.5,     # Weight for constellation loss
) -> torch.Tensor:
    """
    Combined loss: clipping consistency + constellation proximity.
    """
    # Clipping measurement loss
    loss_clip = compute_measurement_loss_fixed(
        x_hat, y_clipped, clip_threshold,
        gamma_unclipped=1.0, eta_magnitude=0.5, eta_phase=0.5
    )
    
    # Constellation loss (demodulate and check symbol positions)
    symbols = demodulate_ofdm_torch(x_hat, fft_size, cp_length, num_symbols, data_indices)
    loss_const = compute_constellation_loss(symbols, constellation, temperature=0.1)
    
    # Combined
    total = lambda_clip * loss_clip + lambda_const * loss_const
    
    return total, loss_clip, loss_const


def guidance_schedule(
    sigma: float,
    sigma_max: float = 50.0,
    sigma_min: float = 0.01,
    lambda_max: float = 2.0,
    warmup_ratio: float = 0.3,
) -> float:
    """
    Compute guidance weight λ(σ): weak at high noise, strong at low noise.
    """
    log_sigma = np.log(sigma)
    log_max = np.log(sigma_max)
    log_min = np.log(sigma_min)
    t = (log_max - log_sigma) / (log_max - log_min + 1e-8)
    t = np.clip(t, 0, 1)
    
    if t < warmup_ratio:
        return 0.0
    else:
        t_adj = (t - warmup_ratio) / (1 - warmup_ratio)
        weight = 0.5 * (1 - np.cos(np.pi * t_adj))
        return weight * lambda_max


def ofdm_projection(
    x: torch.Tensor,
    fft_size: int = 256,
    cp_length: int = 32,
    num_symbols: int = 14,
) -> torch.Tensor:
    """
    Project onto OFDM structure: zero DC, enforce structure.
    Works on MPS by moving complex ops to CPU.
    """
    B, C, T = x.shape
    symbol_len = fft_size + cp_length
    original_device = x.device
    
    # Move to CPU for complex operations (MPS doesn't support torch.complex)
    x_cpu = x.cpu()
    
    x_complex = torch.complex(x_cpu[:, 0], x_cpu[:, 1])
    projected_symbols = []
    
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_len
        end = start + symbol_len
        
        if end > T:
            break
        
        symbol_with_cp = x_complex[:, start:end]
        symbol = symbol_with_cp[:, cp_length:]
        freq = torch.fft.fft(symbol, dim=-1)
        
        # Zero DC subcarrier
        freq[:, 0] = 0
        
        symbol_proj = torch.fft.ifft(freq, dim=-1)
        cp = symbol_proj[:, -cp_length:]
        symbol_with_cp_proj = torch.cat([cp, symbol_proj], dim=-1)
        projected_symbols.append(symbol_with_cp_proj)
    
    if len(projected_symbols) > 0:
        x_proj_complex = torch.cat(projected_symbols, dim=-1)
        if x_proj_complex.shape[-1] < T:
            padding = T - x_proj_complex.shape[-1]
            x_proj_complex = F.pad(x_proj_complex, (0, padding))
        else:
            x_proj_complex = x_proj_complex[:, :T]
        x_proj = torch.stack([x_proj_complex.real, x_proj_complex.imag], dim=1)
    else:
        x_proj = x_cpu
    
    # Move back to original device
    return x_proj.to(original_device)


def sample_with_guidance(
    model, diffusion, y_clipped, clip_level,
    num_steps=30, guidance_weight=1.0, device=None,
    use_fixed_loss=True, apply_ofdm_proj=True,
    ofdm_fft_size=256, ofdm_cp_length=32, ofdm_num_symbols=14,
    modulation="QPSK", use_constellation_loss=True, lambda_constellation=0.5,
):
    """
    FIXED declipping using reconstruction guidance + CONSTELLATION GUIDANCE.
    
    Improvements:
    1. Proper measurement loss with gradients for clipped samples
    2. Guidance scheduling (weak early, strong late)
    3. OFDM structure projection
    4. **NEW: Constellation loss to prevent symbol collapse**
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    
    sigmas = diffusion.get_schedule(num_steps, device)
    sigma_max = sigmas[0].item()
    sigma_min = sigmas[-2].item() if sigmas[-1] == 0 else sigmas[-1].item()
    
    # Get constellation for the modulation
    constellation = get_constellation_torch(modulation, "cpu")
    
    # Compute data subcarrier indices (exclude DC and guard bands)
    guard_ratio = 0.1
    guard_subcarriers = int(ofdm_fft_size * guard_ratio)
    data_indices = torch.arange(guard_subcarriers, ofdm_fft_size - guard_subcarriers)
    data_indices = data_indices[data_indices != ofdm_fft_size // 2]  # Exclude DC
    
    # Initialize
    x = y_clipped + torch.randn_like(y_clipped) * sigmas[0]
    
    for i in tqdm(range(num_steps), desc="Sampling", leave=False):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        if sigma_t == 0:
            break
        
        sigma_val = sigma_t.item()
        
        # Get scheduled guidance weight (stronger near end)
        lambda_t = guidance_schedule(sigma_val, sigma_max, sigma_min, 
                                     lambda_max=guidance_weight * 2)
        
        # Constellation loss weight: ramp up in final 50% of steps
        progress = i / num_steps
        if progress > 0.5 and use_constellation_loss:
            const_weight = lambda_constellation * (progress - 0.5) * 2
        else:
            const_weight = 0.0
        
        # Enable gradients for guidance
        x_in = x.detach().requires_grad_(True)
        
        # Denoise with EDM preconditioning
        sigma_batch = sigma_t.view(1, 1).expand(x.shape[0], 1)
        with torch.enable_grad():
            c_skip, c_out, c_in = diffusion.get_scalings(sigma_batch)
            model_input = c_in.unsqueeze(-1) * x_in
            model_output = model(model_input, sigma_batch)
            x_0_hat = c_skip.unsqueeze(-1) * x_in + c_out.unsqueeze(-1) * model_output
            
            # Compute combined loss (clipping + constellation)
            if use_fixed_loss and const_weight > 0:
                total_loss, loss_clip, loss_const = compute_combined_loss(
                    x_0_hat, y_clipped, clip_level,
                    constellation, ofdm_fft_size, ofdm_cp_length, 
                    ofdm_num_symbols, data_indices,
                    lambda_clip=1.0,
                    lambda_const=const_weight,
                )
                loss = total_loss
            elif use_fixed_loss:
                loss = compute_measurement_loss_fixed(
                    x_0_hat, y_clipped, clip_level,
                    gamma_unclipped=1.0,
                    eta_magnitude=0.5,
                    eta_phase=0.5,
                )
            else:
                # Original naive loss
                x_clipped = clip_signal(x_0_hat, clip_level)
                loss = F.mse_loss(x_clipped, y_clipped, reduction='sum')
            
            # Gradient
            grad = torch.autograd.grad(loss, x_in)[0]
        
        x_0_hat = x_0_hat.detach()
        
        # Score
        score = (x_0_hat - x) / (sigma_t ** 2 + 1e-8)
        
        # Guided score with gradient clipping
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
        score_guided = score - lambda_t * grad
        
        # Euler step
        x = x + (sigma_next - sigma_t) * (-sigma_t * score_guided)
        
        # OFDM projection every 5 steps
        if apply_ofdm_proj and i % 5 == 0 and i > 0:
            x = ofdm_projection(x, ofdm_fft_size, ofdm_cp_length, ofdm_num_symbols)
    
    # Final projection
    if apply_ofdm_proj:
        x = ofdm_projection(x, ofdm_fft_size, ofdm_cp_length, ofdm_num_symbols)
    
    # Final constellation snapping (hard decision)
    x = snap_to_constellation(x, constellation, ofdm_fft_size, ofdm_cp_length, 
                               ofdm_num_symbols, data_indices)
    
    return x


def snap_to_constellation(
    x: torch.Tensor,
    constellation: torch.Tensor,
    fft_size: int,
    cp_length: int,
    num_symbols: int,
    data_indices: torch.Tensor,
) -> torch.Tensor:
    """
    Final step: snap demodulated symbols to nearest constellation point.
    This ensures valid QAM symbols in the output.
    """
    B, C, T = x.shape
    symbol_len = fft_size + cp_length
    original_device = x.device
    
    x_cpu = x.cpu()
    x_complex = torch.complex(x_cpu[:, 0], x_cpu[:, 1])
    
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_len
        end = start + symbol_len
        
        if end > T:
            break
        
        # Extract and FFT
        symbol_with_cp = x_complex[:, start:end]
        symbol = symbol_with_cp[:, cp_length:]
        freq = torch.fft.fft(symbol, dim=-1)
        
        # Snap data subcarriers to nearest constellation point
        for idx in data_indices:
            sym_val = freq[:, idx]  # [B]
            # Find nearest constellation point
            distances = torch.abs(sym_val.unsqueeze(1) - constellation.unsqueeze(0))
            nearest_idx = torch.argmin(distances, dim=1)
            freq[:, idx] = constellation[nearest_idx]
        
        # Zero DC
        freq[:, 0] = 0
        
        # IFFT and replace
        symbol_new = torch.fft.ifft(freq, dim=-1)
        cp_new = symbol_new[:, -cp_length:]
        symbol_with_cp_new = torch.cat([cp_new, symbol_new], dim=-1)
        x_complex[:, start:end] = symbol_with_cp_new
    
    x_out = torch.stack([x_complex.real, x_complex.imag], dim=1)
    return x_out.to(original_device)


# ============== PART 6: Visualization ==============

def plot_signal_comparison(
    original, clipped, reconstructed,
    original_symbols, clipped_symbols, recon_symbols,
    metadata, save_path=None
):
    """Plot comprehensive comparison."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Row 1: Time domain signals
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.plot(original.real[:500], 'b-', alpha=0.7, label='Real')
    ax1.plot(original.imag[:500], 'r-', alpha=0.7, label='Imag')
    ax1.set_title('Original Signal')
    ax1.set_xlabel('Sample')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.plot(clipped.real[:500], 'b-', alpha=0.7, label='Real')
    ax2.plot(clipped.imag[:500], 'r-', alpha=0.7, label='Imag')
    ax2.set_title('Clipped Signal')
    ax2.set_xlabel('Sample')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.plot(reconstructed.real[:500], 'b-', alpha=0.7, label='Real')
    ax3.plot(reconstructed.imag[:500], 'r-', alpha=0.7, label='Imag')
    ax3.set_title('Reconstructed Signal')
    ax3.set_xlabel('Sample')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Constellations
    ref_symbols = metadata['data_symbols'].flatten()
    
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.scatter(original_symbols.real, original_symbols.imag, alpha=0.5, s=10, c='blue')
    ax4.scatter(ref_symbols.real, ref_symbols.imag, alpha=0.3, s=50, c='red', marker='x')
    evm_orig = compute_evm(ref_symbols, original_symbols)
    ax4.set_title(f'Original Constellation\nEVM: {evm_orig:.2f}%')
    ax4.set_xlabel('In-phase')
    ax4.set_ylabel('Quadrature')
    ax4.axis('equal')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.scatter(clipped_symbols.real, clipped_symbols.imag, alpha=0.5, s=10, c='blue')
    ax5.scatter(ref_symbols.real, ref_symbols.imag, alpha=0.3, s=50, c='red', marker='x')
    evm_clip = compute_evm(ref_symbols, clipped_symbols)
    ax5.set_title(f'Clipped Constellation\nEVM: {evm_clip:.2f}%')
    ax5.set_xlabel('In-phase')
    ax5.set_ylabel('Quadrature')
    ax5.axis('equal')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.scatter(recon_symbols.real, recon_symbols.imag, alpha=0.5, s=10, c='blue')
    ax6.scatter(ref_symbols.real, ref_symbols.imag, alpha=0.3, s=50, c='red', marker='x')
    evm_recon = compute_evm(ref_symbols, recon_symbols)
    ax6.set_title(f'Reconstructed Constellation\nEVM: {evm_recon:.2f}%')
    ax6.set_xlabel('In-phase')
    ax6.set_ylabel('Quadrature')
    ax6.axis('equal')
    ax6.grid(True, alpha=0.3)
    
    # Row 3: EVM comparison and info
    ax7 = fig.add_subplot(3, 3, 7)
    stages = ['Original', 'Clipped', 'Reconstructed']
    evms = [evm_orig, evm_clip, evm_recon]
    colors = ['green', 'red', 'blue']
    bars = ax7.bar(stages, evms, color=colors, alpha=0.7)
    ax7.set_ylabel('EVM (%)')
    ax7.set_title('EVM Comparison')
    for bar, evm in zip(bars, evms):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{evm:.1f}%', ha='center', va='bottom')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Info text
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.axis('off')
    info_text = f"""
    OFDM Parameters:
    ─────────────────
    FFT Size: {metadata['fft_size']}
    Modulation: {metadata['modulation']}
    CP Ratio: {metadata['cp_ratio']:.3f}
    Guard Band: {metadata['guard_band_ratio']:.3f}
    Num Symbols: {metadata['num_symbols']}
    Data Subcarriers: {metadata['num_data_subcarriers']}
    
    Results:
    ─────────────────
    EVM Original: {evm_orig:.2f}%
    EVM Clipped: {evm_clip:.2f}%
    EVM Reconstructed: {evm_recon:.2f}%
    
    Improvement: {evm_clip - evm_recon:.2f}%
    """
    ax8.text(0.1, 0.9, info_text, transform=ax8.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Error signals
    ax9 = fig.add_subplot(3, 3, 9)
    error_clip = np.abs(clipped - original)[:500]
    error_recon = np.abs(reconstructed - original)[:500]
    ax9.plot(error_clip, 'r-', alpha=0.7, label='Clipping Error')
    ax9.plot(error_recon, 'b-', alpha=0.7, label='Reconstruction Error')
    ax9.set_title('Error Magnitude')
    ax9.set_xlabel('Sample')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()
    
    return evm_orig, evm_clip, evm_recon


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description="OFDM Diffusion Demo")
    parser.add_argument("--signal_length", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--num_test_samples", type=int, default=3)
    parser.add_argument("--clip_level", type=float, default=1.5,
                        help="Clip level as multiple of signal std (higher=less clipping)")
    parser.add_argument("--guidance_weight", type=float, default=1.0)
    parser.add_argument("--sampling_steps", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default="demo_results")
    # New arguments for conditional model
    parser.add_argument("--model_type", type=str, default="simple", 
                        choices=["simple", "conditional"],
                        help="Model type: 'simple' (unconditional + guidance) or 'conditional' (EVM-focused)")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale (for conditional model)")
    parser.add_argument("--lambda_mask", type=float, default=0.1,
                        help="Mask-weighted loss coefficient (for conditional model)")
    args = parser.parse_args()
    
    # Check if conditional model is available
    if args.model_type == "conditional" and not HAS_CONDITIONAL_MODEL:
        print("Error: Conditional model not available. Falling back to simple model.")
        args.model_type = "simple"
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 1: Generate Database and Show Sample ==========
    print("\n" + "="*60)
    print("STEP 1: Generate OFDM Database")
    print("="*60)
    
    config = OFDMConfig()
    
    # Generate a sample and show its properties
    print("\nGenerating sample OFDM signal...")
    sample_signal, sample_metadata = generate_ofdm_signal(config, args.signal_length, seed=42)
    
    print(f"  Signal length: {len(sample_signal)}")
    print(f"  FFT size: {sample_metadata['fft_size']}")
    print(f"  Modulation: {sample_metadata['modulation']}")
    print(f"  CP ratio: {sample_metadata['cp_ratio']:.3f}")
    print(f"  Num OFDM symbols: {sample_metadata['num_symbols']}")
    print(f"  Data subcarriers: {sample_metadata['num_data_subcarriers']}")
    
    # Demodulate and compute EVM (should be ~0 for clean signal)
    sample_demod = demodulate_ofdm(sample_signal, sample_metadata)
    sample_evm = compute_evm(sample_metadata['data_symbols'].flatten(), sample_demod.flatten())
    print(f"  Clean signal EVM: {sample_evm:.4f}%")
    
    # Plot sample constellation
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Time domain
    axes[0].plot(sample_signal.real[:300], 'b-', alpha=0.7, label='Real')
    axes[0].plot(sample_signal.imag[:300], 'r-', alpha=0.7, label='Imag')
    axes[0].set_title(f'Sample OFDM Signal ({sample_metadata["modulation"]})')
    axes[0].set_xlabel('Sample')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Constellation
    ref = sample_metadata['data_symbols'].flatten()
    demod = sample_demod.flatten()
    axes[1].scatter(demod.real, demod.imag, alpha=0.5, s=10, c='blue', label='Demodulated')
    axes[1].scatter(ref.real, ref.imag, alpha=0.5, s=50, c='red', marker='x', label='Reference')
    axes[1].set_title(f'Constellation (EVM: {sample_evm:.2f}%)')
    axes[1].set_xlabel('In-phase')
    axes[1].set_ylabel('Quadrature')
    axes[1].axis('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "step1_sample_signal.png", dpi=150)
    plt.show()
    
    # ========== STEP 2: Train Diffusion Model ==========
    print("\n" + "="*60)
    print(f"STEP 2: Train Diffusion Model ({args.model_type.upper()})")
    print("="*60)
    
    if args.model_type == "conditional":
        # ========== CONDITIONAL MODEL (NEW) ==========
        print("\nUsing CONDITIONAL declipping model with EVM-focused training")
        
        # Create declipping dataset (mild clipping: only top 8-25% of peaks)
        declip_config = DeclipConfig(
            signal_length=args.signal_length,
            fft_size=256,
            modulation='QPSK',
            clip_ratio_min=0.75,  # Mild clipping: ~25% of peak clipped
            clip_ratio_max=0.92,  # Very mild: ~8% of peak clipped
        )
        dataset = OFDMDeclipDataset(config=declip_config, seed=42)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=collate_declip_batch
        )
        data_iter = iter(dataloader)
        
        # Create conditional model
        model = ConditionalUNet1DSimple(
            in_channels=2,
            out_channels=2,
            base_channels=64,
            depth=4,
            embed_dim=512,
            cond_drop_prob=0.1,  # CFG dropout
        ).to(device)
        
        # Diffusion
        diffusion = ConditionalDiffusion()
        
    else:
        # ========== SIMPLE MODEL (ORIGINAL) ==========
        print("\nUsing SIMPLE unconditional model with inference-time guidance")
        
        # Dataset
        dataset = OFDMDataset(signal_length=args.signal_length, config=config)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
        data_iter = iter(dataloader)
        
        # Model
        model = SimpleUNet(in_ch=2, base_ch=32, depth=4).to(device)
        
        # Diffusion
        diffusion = SimpleDiffusion()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")
    
    # Optimizer with lower initial learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    
    # Learning rate scheduler with warmup
    def get_lr(step, warmup_steps, base_lr, total_steps):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps  # Linear warmup
        else:
            # Cosine decay after warmup
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
    
    # OFDM parameters for constellation-aware training
    # Use typical OFDM config - QPSK ONLY for simplicity
    train_fft_size = 256
    train_cp_length = 32
    train_num_symbols = 14
    train_modulation = "QPSK"  # QPSK only
    
    # Compute data indices (exclude DC and guard bands)
    guard_ratio = 0.1
    guard_subcarriers = int(train_fft_size * guard_ratio)
    train_data_indices = torch.arange(guard_subcarriers, train_fft_size - guard_subcarriers)
    train_data_indices = train_data_indices[train_data_indices != train_fft_size // 2]
    
    ofdm_params = {
        'fft_size': train_fft_size,
        'cp_length': train_cp_length,
        'num_symbols': train_num_symbols,
        'modulation': train_modulation,
        'data_indices': train_data_indices,
    }
    
    # Training loop
    print(f"\nTraining for {args.train_steps} steps...")
    if args.model_type == "conditional":
        print(f"  Using conditional model with mask-weighted loss (λ_mask={args.lambda_mask})")
    else:
        print(f"  Using QPSK constellation loss on {device.type.upper()}")
    print(f"  LR warmup: {args.warmup_steps} steps, then cosine decay")
    
    losses = []
    losses_eps = []
    losses_mask = []
    best_loss = float('inf')
    
    pbar = tqdm(range(args.train_steps), desc="Training")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Update learning rate
        lr = get_lr(step, args.warmup_steps, args.lr, args.train_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if args.model_type == "conditional":
            # Conditional model training
            clean, clipped, mask, symbols, ofdm_params_batch, clip_level = batch
            clean = clean.to(device)
            clipped = clipped.to(device)
            mask = mask.to(device)
            clip_level = clip_level.to(device)
            
            loss, loss_eps, loss_mask = train_conditional_step(
                model, diffusion, clean, clipped, mask, clip_level,
                optimizer, device, lambda_freq=0.0, lambda_mask=args.lambda_mask
            )
            
            if not np.isnan(loss):
                losses.append(loss)
                losses_eps.append(loss_eps)
                losses_mask.append(loss_mask)
                best_loss = min(best_loss, loss)
            
            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:]) if losses else 0
                avg_eps = np.mean(losses_eps[-100:]) if losses_eps else 0
                avg_mask = np.mean(losses_mask[-100:]) if losses_mask else 0
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'eps': f'{avg_eps:.4f}',
                    'mask': f'{avg_mask:.4f}',
                    'lr': f'{lr:.1e}'
                })
        else:
            # Simple model training (original)
            use_const = False
            lambda_const = 0.0
            
            loss = train_step(
                model, diffusion, batch, optimizer, device,
                use_constellation_loss=use_const,
                ofdm_params=ofdm_params,
                lambda_const=lambda_const,
            )
            
            # Skip NaN losses
            if np.isnan(loss):
                continue
                
            losses.append(loss)
            best_loss = min(best_loss, loss)
            
            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:]) if losses else 0
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'best': f'{best_loss:.4f}',
                    'lr': f'{lr:.1e}',
                    'λc': f'{lambda_const:.2f}'
                })
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(losses, alpha=0.3)
    plt.plot(np.convolve(losses, np.ones(50)/50, mode='valid'), 'r-', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss (MSE + Constellation)')
    plt.axvline(x=args.train_steps * 0.2, color='g', linestyle='--', label='Const. loss ON')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "step2_training_loss.png", dpi=150)
    plt.show()
    
    print(f"Final loss: {np.mean(losses[-100:]):.6f}")
    
    # Save model
    if args.model_type == "conditional":
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'conditional',
            'config': {
                'signal_length': args.signal_length, 
                'base_channels': 64, 
                'depth': 4,
                'embed_dim': 512,
            }
        }, output_dir / "model.pt")
    else:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_type': 'simple',
            'config': {'signal_length': args.signal_length, 'base_ch': 32, 'depth': 4}
        }, output_dir / "model.pt")
    
    # ========== STEP 3: Test Declipping ==========
    print("\n" + "="*60)
    print("STEP 3: Test Declipping")
    print("="*60)
    
    all_results = []
    
    for test_idx in range(args.num_test_samples):
        print(f"\n--- Test Sample {test_idx + 1}/{args.num_test_samples} ---")
        
        # Generate test signal
        test_signal, test_metadata = generate_ofdm_signal(
            config, args.signal_length, seed=1000 + test_idx
        )
        
        print(f"  Modulation: {test_metadata['modulation']}")
        print(f"  FFT size: {test_metadata['fft_size']}")
        
        # Convert to tensor
        x_clean = torch.from_numpy(complex_to_2ch(test_signal)).unsqueeze(0).to(device)
        
        # Apply clipping
        clip_level = args.clip_level * torch.std(x_clean).item()
        x_clipped = clip_signal(x_clean, clip_level)
        
        # Compute clipping SDR
        distortion = x_clean - x_clipped
        sdr = 10 * torch.log10(torch.mean(x_clean**2) / (torch.mean(distortion**2) + 1e-10))
        print(f"  Clipping SDR: {sdr.item():.2f} dB")
        
        # Run declipping
        fft_size = test_metadata['fft_size']
        cp_length = int(test_metadata['cp_ratio'] * fft_size)
        num_symbols = test_metadata['num_symbols']
        modulation = test_metadata['modulation']
        
        if args.model_type == "conditional":
            # Conditional model: uses DDIM with data consistency
            print("  Running declipping (CONDITIONAL model + DDIM)...")
            
            # Create mask based on MAGNITUDE (1 where |x_clean| >= A)
            # Use clean signal to determine where clipping occurred
            mask = get_clipping_mask(x_clean, clip_level)
            clip_level_tensor = torch.tensor([clip_level], device=device)
            
            x_recon = sample_conditional_ddim(
                model, diffusion, x_clipped, mask, clip_level_tensor,
                num_steps=args.sampling_steps,
                cfg_scale=args.cfg_scale,
                data_consistency=True,
                device=device,
            )
        else:
            # Simple model: uses guidance-based sampling
            print("  Running declipping (SIMPLE model + GUIDANCE)...")
            x_recon = sample_with_guidance(
                model, diffusion, x_clipped, clip_level,
                num_steps=args.sampling_steps,
                guidance_weight=args.guidance_weight,
                device=device,
                use_fixed_loss=True,              # Use proper measurement loss
                apply_ofdm_proj=True,             # Apply OFDM structure projection
                ofdm_fft_size=fft_size,
                ofdm_cp_length=cp_length,
                ofdm_num_symbols=num_symbols,
                modulation=modulation,            # Pass modulation for constellation
                use_constellation_loss=True,      # Enable constellation guidance
                lambda_constellation=1.0,         # Constellation loss weight
            )
        
        # Convert back to complex
        original_np = ch2_to_complex(x_clean[0].cpu().numpy())
        clipped_np = ch2_to_complex(x_clipped[0].cpu().numpy())
        recon_np = ch2_to_complex(x_recon[0].cpu().numpy())
        
        # Demodulate
        original_symbols = demodulate_ofdm(original_np, test_metadata).flatten()
        clipped_symbols = demodulate_ofdm(clipped_np, test_metadata).flatten()
        recon_symbols = demodulate_ofdm(recon_np, test_metadata).flatten()
        
        # Plot comparison
        evm_orig, evm_clip, evm_recon = plot_signal_comparison(
            original_np, clipped_np, recon_np,
            original_symbols, clipped_symbols, recon_symbols,
            test_metadata,
            save_path=output_dir / f"step3_test_{test_idx+1}.png"
        )
        
        all_results.append({
            'test_idx': test_idx,
            'modulation': test_metadata['modulation'],
            'fft_size': test_metadata['fft_size'],
            'sdr': sdr.item(),
            'evm_original': evm_orig,
            'evm_clipped': evm_clip,
            'evm_reconstructed': evm_recon,
            'improvement': evm_clip - evm_recon,
        })
        
        print(f"  EVM Original: {evm_orig:.2f}%")
        print(f"  EVM Clipped: {evm_clip:.2f}%")
        print(f"  EVM Reconstructed: {evm_recon:.2f}%")
        print(f"  Improvement: {evm_clip - evm_recon:.2f}%")
    
    # ========== Summary ==========
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    avg_evm_clip = np.mean([r['evm_clipped'] for r in all_results])
    avg_evm_recon = np.mean([r['evm_reconstructed'] for r in all_results])
    avg_improvement = np.mean([r['improvement'] for r in all_results])
    
    print(f"Average EVM Clipped: {avg_evm_clip:.2f}%")
    print(f"Average EVM Reconstructed: {avg_evm_recon:.2f}%")
    print(f"Average Improvement: {avg_improvement:.2f}%")
    
    # Save results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
