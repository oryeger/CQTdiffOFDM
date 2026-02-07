#!/usr/bin/env python3
"""
Training Script for Conditional OFDM Declipping Diffusion Model.

Trains a conditional diffusion model that reconstructs clean complex OFDM
time-domain IQ from clipped observations, optimized for low EVM after FFT.

Features:
- Conditional U-Net with (y, m, A) conditioning
- EVM-focused frequency loss
- Mask-weighted time loss for clipped regions
- DDIM sampling with data consistency (inpainting)
- Classifier-free guidance support

Usage:
    python train_ofdm_declip.py --signal_length 8192 --max_steps 100000
    python train_ofdm_declip.py --config experiments/declip_config.yaml
"""

import os
import argparse
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from src.models.unet_ofdm_declip import ConditionalUNet1DSimple
from src.ofdm.ofdm_declip_dataset import (
    DeclipConfig,
    OFDMDeclipDataset,
    OFDMDeclipDatasetFixed,
    create_declip_dataloader,
    create_test_dataloader,
    collate_declip_batch,
    compute_sdr,
)
from src.ofdm.ofdm_generator import OFDMParams, demodulate_ofdm_torch
from src.utils.evm import compute_evm
from src.sampler_declip import DDIMSamplerDeclip, create_ddim_sampler


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    base_channels: int = 64
    depth: int = 4
    embed_dim: int = 512
    cond_drop_prob: float = 0.1
    
    # Data
    signal_length: int = 8192
    fft_size: int = 64
    modulation: str = '16QAM'
    clip_ratio_min: float = 0.3
    clip_ratio_max: float = 0.7
    
    # Training
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_steps: int = 100000
    warmup_steps: int = 1000
    
    # Loss weights
    lambda_freq: float = 0.1
    lambda_mask: float = 0.1
    
    # Diffusion
    sigma_min: float = 1e-4
    sigma_max: float = 10.0
    sigma_data: float = 1.0
    rho: float = 7.0
    
    # Logging & checkpointing
    log_interval: int = 100
    save_interval: int = 10000
    eval_interval: int = 5000
    output_dir: str = "experiments/ofdm_declip"
    
    # Sampling
    num_sample_steps: int = 50
    cfg_scale: float = 1.5


class EDMDiffusion:
    """
    EDM-style diffusion (Karras et al.).
    
    Simplified version for declipping training.
    """
    
    def __init__(
        self,
        sigma_min: float = 1e-4,
        sigma_max: float = 10.0,
        sigma_data: float = 1.0,
        rho: float = 7.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
    
    def get_scalings(self, sigma: torch.Tensor):
        """Get c_skip, c_out, c_in, c_noise for EDM parameterization."""
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = torch.log(sigma) / 4
        return c_skip, c_out, c_in, c_noise
    
    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample sigma for training (log-normal distribution)."""
        log_sigma = torch.randn(batch_size, device=device) * 1.2 - 1.2
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)
        return sigma
    
    def get_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Get sigma schedule for sampling."""
        step_indices = torch.arange(num_steps + 1, device=device)
        t = step_indices / num_steps
        
        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        
        sigmas = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** self.rho
        sigmas[-1] = 0
        
        return sigmas
    
    def add_noise(self, x: torch.Tensor, sigma: torch.Tensor) -> tuple:
        """Add noise: x_noisy = x + sigma * noise"""
        noise = torch.randn_like(x)
        if sigma.ndim == 1:
            sigma = sigma.view(-1, 1, 1)
        x_noisy = x + sigma * noise
        return x_noisy, noise


def compute_freq_loss(
    x_pred: torch.Tensor,
    x_clean: torch.Tensor,
    ofdm_params: OFDMParams,
    num_symbols: int,
) -> torch.Tensor:
    """
    Compute frequency-domain loss (EVM-related).
    
    Demodulates signals and computes MSE on constellation points.
    """
    # Convert 2-channel to complex
    x_pred_complex = torch.complex(x_pred[:, 0, :], x_pred[:, 1, :])
    x_clean_complex = torch.complex(x_clean[:, 0, :], x_clean[:, 1, :])
    
    # Demodulate
    X_pred = demodulate_ofdm_torch(x_pred_complex, ofdm_params, num_symbols)
    X_clean = demodulate_ofdm_torch(x_clean_complex, ofdm_params, num_symbols)
    
    # MSE on constellation points
    error = X_pred - X_clean
    mse = torch.mean(torch.abs(error) ** 2)
    
    # Normalize by reference power
    ref_power = torch.mean(torch.abs(X_clean) ** 2) + 1e-8
    return mse / ref_power


def compute_mask_loss(
    x_pred: torch.Tensor,
    x_clean: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute mask-weighted time loss.
    
    Emphasizes clipped regions (where mask=1).
    """
    error = x_pred - x_clean
    
    # Expand mask
    if mask.ndim == 3 and mask.shape[1] == 1:
        mask = mask.expand(-1, x_pred.shape[1], -1)
    
    weighted_error = mask * (error ** 2)
    
    # Normalize by number of clipped samples
    num_clipped = mask.sum() + 1e-8
    return weighted_error.sum() / num_clipped


def train_step(
    model: nn.Module,
    diffusion: EDMDiffusion,
    batch: tuple,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    device: torch.device,
) -> dict:
    """Single training step with composite loss."""
    model.train()
    optimizer.zero_grad()
    
    # Unpack batch
    clean, clipped, mask, symbols, ofdm_params, clip_level = batch
    clean = clean.to(device)
    clipped = clipped.to(device)
    mask = mask.to(device)
    clip_level = clip_level.to(device)
    
    B = clean.shape[0]
    
    # Sample noise level
    sigma = diffusion.sample_sigma(B, device)
    
    # Add noise to clean signal
    x_noisy, noise = diffusion.add_noise(clean, sigma)
    
    # Get scalings
    c_skip, c_out, c_in, c_noise = diffusion.get_scalings(sigma)
    c_in = c_in.view(-1, 1, 1)
    c_noise = c_noise.view(-1, 1)
    
    # Forward pass
    eps_pred = model(
        x_t=c_in * x_noisy,
        t=c_noise.squeeze(-1),
        y=clipped,
        m=mask,
        A=clip_level.squeeze(-1),
    )
    
    # Compute target (noise prediction parameterization)
    # model predicts: eps such that x_0 = c_skip * x_t + c_out * (x_0 - c_skip*x_t)/c_out
    # Simplified: model output -> eps, target = (x_0 - c_skip*x_t) / c_out
    c_skip = c_skip.view(-1, 1, 1)
    c_out = c_out.view(-1, 1, 1)
    target = (clean - c_skip * x_noisy) / (c_out + 1e-8)
    
    # ========== LOSSES ==========
    
    # 1. Primary noise prediction loss
    loss_eps = torch.mean((eps_pred - target) ** 2)
    
    # 2. Frequency loss (for low noise levels)
    sigma_threshold = 0.5
    low_noise_mask = sigma < sigma_threshold
    
    if low_noise_mask.any() and config.lambda_freq > 0 and ofdm_params is not None:
        # Reconstruct x_0 from prediction
        x_pred = c_skip * x_noisy + c_out * eps_pred
        
        # Compute frequency loss on low-noise samples
        idx = low_noise_mask.nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            num_symbols = symbols.shape[0] if symbols is not None else ofdm_params.num_data_subcarriers
            loss_freq = compute_freq_loss(
                x_pred[idx], clean[idx], ofdm_params, num_symbols
            )
        else:
            loss_freq = torch.tensor(0.0, device=device)
    else:
        loss_freq = torch.tensor(0.0, device=device)
    
    # 3. Mask-weighted time loss
    if config.lambda_mask > 0:
        x_pred = c_skip * x_noisy + c_out * eps_pred
        loss_mask = compute_mask_loss(x_pred, clean, mask)
    else:
        loss_mask = torch.tensor(0.0, device=device)
    
    # Total loss
    total_loss = loss_eps + config.lambda_freq * loss_freq + config.lambda_mask * loss_mask
    
    # Backward
    total_loss.backward()
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Optimizer step
    optimizer.step()
    
    return {
        'total': total_loss.item(),
        'eps': loss_eps.item(),
        'freq': loss_freq.item() if isinstance(loss_freq, torch.Tensor) else loss_freq,
        'mask': loss_mask.item() if isinstance(loss_mask, torch.Tensor) else loss_mask,
        'grad_norm': grad_norm.item(),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    diffusion: EDMDiffusion,
    test_loader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
    num_samples: int = 10,
) -> dict:
    """Evaluate model on test set."""
    model.eval()
    
    # Create simple DDIM sampler (without VE_Sde wrapper)
    results = {
        'sdr': [],
        'evm': [],
    }
    
    sigmas = diffusion.get_schedule(config.num_sample_steps, device)
    
    for i, batch in enumerate(test_loader):
        if i >= num_samples:
            break
        
        clean, clipped, mask, symbols, ofdm_params, clip_level = batch
        clean = clean.to(device)
        clipped = clipped.to(device)
        mask = mask.to(device)
        clip_level = clip_level.to(device)
        
        B, C, T = clean.shape
        
        # Initialize from noise
        x = torch.randn(B, C, T, device=device) * sigmas[0]
        
        # DDIM sampling with data consistency
        for j in range(config.num_sample_steps):
            t_curr = sigmas[j]
            t_next = sigmas[j + 1]
            
            c_skip, c_out, c_in, c_noise = diffusion.get_scalings(t_curr.view(1))
            c_in = c_in.view(1, 1, 1)
            c_noise = c_noise.squeeze()
            
            # Denoise
            eps = model(
                x_t=c_in * x,
                t=c_noise.expand(B),
                y=clipped,
                m=mask,
                A=clip_level.squeeze(-1),
            )
            
            x_0 = c_skip.view(1, 1, 1) * x + c_out.view(1, 1, 1) * eps
            
            # Data consistency (inpainting)
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
        
        # Compute metrics
        reconstructed = x
        
        # SDR
        sdr = compute_sdr(clean, reconstructed)
        results['sdr'].append(sdr)
        
        # EVM (if symbols available)
        if symbols is not None and ofdm_params is not None:
            recon_complex = torch.complex(reconstructed[:, 0, :], reconstructed[:, 1, :])
            clean_complex = torch.complex(clean[:, 0, :], clean[:, 1, :])
            
            num_syms = symbols.shape[0]
            recovered = demodulate_ofdm_torch(recon_complex[0], ofdm_params, num_syms)
            original = demodulate_ofdm_torch(clean_complex[0], ofdm_params, num_syms)
            
            evm = compute_evm(original.cpu().numpy(), recovered.cpu().numpy())
            results['evm'].append(evm)
    
    return {
        'sdr_mean': np.mean(results['sdr']),
        'sdr_std': np.std(results['sdr']),
        'evm_mean': np.mean(results['evm']) if results['evm'] else 0,
        'evm_std': np.std(results['evm']) if results['evm'] else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train OFDM Declipping Model")
    
    # Model
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--cond_drop_prob", type=float, default=0.1)
    
    # Data
    parser.add_argument("--signal_length", type=int, default=8192)
    parser.add_argument("--fft_size", type=int, default=64)
    parser.add_argument("--modulation", type=str, default="16QAM")
    parser.add_argument("--clip_ratio_min", type=float, default=0.3)
    parser.add_argument("--clip_ratio_max", type=float, default=0.7)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100000)
    
    # Loss weights
    parser.add_argument("--lambda_freq", type=float, default=0.1)
    parser.add_argument("--lambda_mask", type=float, default=0.1)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--output_dir", type=str, default="experiments/ofdm_declip")
    
    # Config file
    parser.add_argument("--config", type=str, default=None)
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            file_config = yaml.safe_load(f)
        for k, v in file_config.items():
            if hasattr(args, k):
                setattr(args, k, v)
    
    config = TrainingConfig(**{k: v for k, v in vars(args).items() if hasattr(TrainingConfig, k)})
    
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(config), f)
    
    # Dataset
    data_config = DeclipConfig(
        signal_length=config.signal_length,
        fft_size=config.fft_size,
        modulation=config.modulation,
        clip_ratio_min=config.clip_ratio_min,
        clip_ratio_max=config.clip_ratio_max,
    )
    
    train_loader = create_declip_dataloader(
        data_config, batch_size=config.batch_size, num_workers=0
    )
    train_iter = iter(train_loader)
    
    test_loader = create_test_dataloader(
        data_config, num_samples=50, batch_size=1
    )
    
    # Model
    model = ConditionalUNet1DSimple(
        in_channels=2,
        out_channels=2,
        base_channels=config.base_channels,
        depth=config.depth,
        embed_dim=config.embed_dim,
        cond_drop_prob=config.cond_drop_prob,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Diffusion
    diffusion = EDMDiffusion(
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        sigma_data=config.sigma_data,
        rho=config.rho,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_steps, eta_min=1e-6
    )
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {config.max_steps} steps...")
    print(f"Loss weights: λ_freq={config.lambda_freq}, λ_mask={config.lambda_mask}")
    print(f"{'='*60}\n")
    
    losses = []
    best_sdr = -float('inf')
    
    for step in range(1, config.max_steps + 1):
        start_time = time.time()
        
        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        
        # Train step
        metrics = train_step(model, diffusion, batch, optimizer, config, device)
        scheduler.step()
        
        losses.append(metrics['total'])
        elapsed = time.time() - start_time
        
        # Logging
        if step % config.log_interval == 0:
            avg_loss = np.mean(losses[-config.log_interval:])
            print(
                f"Step {step:6d}/{config.max_steps} | "
                f"Loss: {avg_loss:.5f} "
                f"(ε={metrics['eps']:.5f}, freq={metrics['freq']:.5f}, mask={metrics['mask']:.5f}) | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.2f}s"
            )
        
        # Evaluation
        if step % config.eval_interval == 0:
            print("\nEvaluating...")
            eval_metrics = evaluate(model, diffusion, test_loader, config, device, num_samples=10)
            print(
                f"  SDR: {eval_metrics['sdr_mean']:.2f} ± {eval_metrics['sdr_std']:.2f} dB | "
                f"EVM: {eval_metrics['evm_mean']:.2f} ± {eval_metrics['evm_std']:.2f} %"
            )
            
            if eval_metrics['sdr_mean'] > best_sdr:
                best_sdr = eval_metrics['sdr_mean']
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': asdict(config),
                    'metrics': eval_metrics,
                }, output_dir / "best_model.pt")
                print(f"  → New best model saved! (SDR: {best_sdr:.2f} dB)")
            print()
        
        # Checkpointing
        if step % config.save_interval == 0:
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'config': asdict(config),
            }, output_dir / f"checkpoint_{step}.pt")
            print(f"Saved checkpoint at step {step}")
    
    # Final save
    torch.save({
        'step': config.max_steps,
        'model_state_dict': model.state_dict(),
        'losses': losses,
        'config': asdict(config),
    }, output_dir / "final_model.pt")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best SDR: {best_sdr:.2f} dB")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
