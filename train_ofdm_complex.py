"""
Training script for Complex Baseband OFDM Diffusion Model.

This trains an unconditional diffusion prior on clean OFDM signals.
The model learns P(x_0) where x_0 = [Re(x), Im(x)] is a 2-channel representation.

Usage:
    python train_ofdm_complex.py --signal_length 8192 --max_steps 100000
"""

import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from src.ofdm.ofdm_complex_generator import OFDMComplexDataset, OFDMConfig
from src.models.unet_ofdm_complex import UNet1DComplexSimple


# ============== EDM-Style Diffusion Parameters ==============

class EDMDiffusion:
    """
    EDM-style diffusion (Karras et al. "Elucidating the Design Space...")
    
    Key equations:
    - Forward: x_sigma = x_0 + sigma * epsilon
    - Denoiser: D(x_sigma, sigma) predicts x_0
    - Training target: x_0 (clean signal)
    """
    
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,  # Estimated std of data
        rho: float = 7.0,  # Schedule parameter
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
    
    def get_scalings(self, sigma: torch.Tensor):
        """
        Get c_skip, c_out, c_in for EDM parameterization.
        
        D(x, sigma) = c_skip * x + c_out * F(c_in * x, c_noise)
        """
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = torch.log(sigma) / 4  # Or use sigma directly
        
        return c_skip, c_out, c_in, c_noise
    
    def sample_sigma_training(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample sigma for training (log-normal distribution)."""
        log_sigma = torch.randn(batch_size, device=device) * 1.2 - 1.2
        sigma = torch.exp(log_sigma)
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)
        return sigma.unsqueeze(-1)  # (B, 1)
    
    def get_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Get sigma schedule for sampling."""
        step_indices = torch.arange(num_steps + 1, device=device)
        t = step_indices / num_steps
        
        sigma_max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigma_min_inv_rho = self.sigma_min ** (1 / self.rho)
        
        sigmas = (sigma_max_inv_rho + t * (sigma_min_inv_rho - sigma_max_inv_rho)) ** self.rho
        sigmas[-1] = 0  # Final step has sigma=0
        
        return sigmas
    
    def add_noise(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Add noise: x_sigma = x + sigma * epsilon"""
        noise = torch.randn_like(x)
        return x + sigma.unsqueeze(-1) * noise, noise


def train_step(
    model: nn.Module,
    diffusion: EDMDiffusion,
    batch: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Single training step.
    
    Args:
        model: Denoiser network
        diffusion: Diffusion parameters
        batch: Clean signals, shape (B, 2, T)
        optimizer: Optimizer
        device: Device
        
    Returns:
        Loss value
    """
    model.train()
    optimizer.zero_grad()
    
    x_0 = batch.to(device)  # Clean signal
    batch_size = x_0.shape[0]
    
    # Sample noise level
    sigma = diffusion.sample_sigma_training(batch_size, device)
    
    # Get EDM scalings
    c_skip, c_out, c_in, c_noise = diffusion.get_scalings(sigma)
    
    # Add noise
    x_sigma, noise = diffusion.add_noise(x_0, sigma)
    
    # Forward pass through model
    # Model predicts F(c_in * x_sigma, c_noise)
    # Full denoiser: D = c_skip * x_sigma + c_out * F(...)
    
    scaled_input = c_in.unsqueeze(-1) * x_sigma
    model_output = model(scaled_input, c_noise)
    
    # Target for model output (not the full denoiser)
    # D = c_skip * x_sigma + c_out * model_output = x_0
    # => model_output = (x_0 - c_skip * x_sigma) / c_out
    target = (x_0 - c_skip.unsqueeze(-1) * x_sigma) / c_out.unsqueeze(-1)
    
    # MSE loss
    loss = torch.mean((model_output - target) ** 2)
    
    # Weight by sigma (optional, can help with different noise levels)
    # Karras uses: weight = (sigma^2 + sigma_data^2) / (sigma * sigma_data)^2
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    return loss.item()


def get_denoised(
    model: nn.Module,
    diffusion: EDMDiffusion,
    x_sigma: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """
    Apply denoiser to get x_0 estimate.
    
    D(x_sigma, sigma) = c_skip * x_sigma + c_out * F(c_in * x_sigma, c_noise)
    """
    c_skip, c_out, c_in, c_noise = diffusion.get_scalings(sigma)
    
    scaled_input = c_in.unsqueeze(-1) * x_sigma
    model_output = model(scaled_input, c_noise)
    
    x_0_hat = c_skip.unsqueeze(-1) * x_sigma + c_out.unsqueeze(-1) * model_output
    
    return x_0_hat


def main():
    parser = argparse.ArgumentParser(description="Train OFDM Diffusion Model")
    parser.add_argument("--signal_length", type=int, default=8192, help="Signal length")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100000, help="Max training steps")
    parser.add_argument("--save_interval", type=int, default=10000, help="Checkpoint interval")
    parser.add_argument("--log_interval", type=int, default=100, help="Log interval")
    parser.add_argument("--output_dir", type=str, default="experiments/ofdm_complex")
    parser.add_argument("--base_channels", type=int, default=64, help="Base channels")
    parser.add_argument("--depth", type=int, default=5, help="U-Net depth")
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset
    config = OFDMConfig()
    dataset = OFDMComplexDataset(signal_length=args.signal_length, config=config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    data_iter = iter(dataloader)
    
    # Model
    model = UNet1DComplexSimple(
        in_channels=2,
        base_channels=args.base_channels,
        depth=args.depth,
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Diffusion
    diffusion = EDMDiffusion(sigma_data=1.0)  # Data is normalized to unit variance
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=1e-6
    )
    
    # Training loop
    print(f"\nStarting training for {args.max_steps} steps...")
    losses = []
    
    for step in range(1, args.max_steps + 1):
        start_time = time.time()
        
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Training step
        loss = train_step(model, diffusion, batch, optimizer, device)
        scheduler.step()
        
        losses.append(loss)
        elapsed = time.time() - start_time
        
        # Logging
        if step % args.log_interval == 0:
            avg_loss = np.mean(losses[-args.log_interval:])
            print(f"Step {step}/{args.max_steps} | Loss: {avg_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e} | Time: {elapsed:.2f}s")
        
        # Save checkpoint
        if step % args.save_interval == 0:
            checkpoint = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "losses": losses,
                "config": {
                    "signal_length": args.signal_length,
                    "base_channels": args.base_channels,
                    "depth": args.depth,
                }
            }
            torch.save(checkpoint, output_dir / f"checkpoint_{step}.pt")
            print(f"Saved checkpoint at step {step}")
    
    # Final save
    torch.save(checkpoint, output_dir / "checkpoint_final.pt")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
