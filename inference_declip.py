"""
Inference script for OFDM Declipping using Reconstruction Guidance.

This implements the declipping algorithm:
1. Start with clipped observation y = clip(x_0)
2. Run reverse diffusion with reconstruction guidance
3. At each step, guide towards clip(x̂_0) ≈ y

Algorithm (Pseudocode):
    x_T ~ N(0, sigma_max^2 * I)
    for t = T, T-1, ..., 1:
        # Denoise to get x_0 estimate
        x̂_0 = D(x_t, sigma_t)
        
        # Reconstruction guidance
        L_meas = ||y - clip(x̂_0)||^2
        grad = ∇_{x_t} L_meas
        
        # Guided score
        score_guided = score(x_t) - lambda * grad
        
        # ODE/SDE step
        x_{t-1} = x_t + (sigma_{t-1} - sigma_t) * score_guided + noise
    
    return x_0

Usage:
    python inference_declip.py --checkpoint experiments/ofdm_complex/checkpoint_final.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from src.ofdm.ofdm_complex_generator import (
    OFDMConfig, generate_ofdm_signal_complex, complex_to_2channel,
    channel_2_to_complex, demodulate_ofdm_complex, compute_evm_complex
)
from src.models.unet_ofdm_complex import UNet1DComplexSimple
from train_ofdm_complex import EDMDiffusion, get_denoised


# ============== Clipping Operators ==============

def clip_signal(x: torch.Tensor, clip_level: float) -> torch.Tensor:
    """
    Apply soft or hard clipping.
    
    Args:
        x: Signal, shape (B, 2, T) - 2 channels [Re, Im]
        clip_level: Clipping threshold (clips both real and imag independently)
        
    Returns:
        Clipped signal
    """
    return torch.clamp(x, -clip_level, clip_level)


def compute_clip_sdr(x_clean: torch.Tensor, x_clipped: torch.Tensor) -> float:
    """Compute SDR (Signal-to-Distortion Ratio) from clipping."""
    distortion = x_clean - x_clipped
    signal_power = torch.mean(x_clean ** 2)
    distortion_power = torch.mean(distortion ** 2) + 1e-10
    sdr = 10 * torch.log10(signal_power / distortion_power)
    return sdr.item()


# ============== Reconstruction Guidance ==============

def reconstruction_guidance_step(
    model: torch.nn.Module,
    diffusion: EDMDiffusion,
    x_t: torch.Tensor,
    sigma_t: torch.Tensor,
    y_clipped: torch.Tensor,
    clip_level: float,
    guidance_weight: float,
    gradient_clip: float = 1.0,
) -> tuple:
    """
    Compute reconstruction guidance gradient.
    
    Args:
        model: Denoiser
        diffusion: Diffusion parameters
        x_t: Current noisy sample, shape (B, 2, T)
        sigma_t: Current noise level
        y_clipped: Observed clipped signal
        clip_level: Clipping threshold
        guidance_weight: Lambda for guidance strength
        gradient_clip: Clip gradient norm
        
    Returns:
        x_0_hat: Denoised estimate
        guidance_grad: Gradient for guidance
    """
    # Enable gradient computation
    x_t = x_t.detach().requires_grad_(True)
    
    # Get denoised estimate
    x_0_hat = get_denoised(model, diffusion, x_t, sigma_t)
    
    # Measurement loss: ||y - clip(x̂_0)||^2
    x_0_clipped = clip_signal(x_0_hat, clip_level)
    measurement_loss = F.mse_loss(x_0_clipped, y_clipped, reduction='sum')
    
    # Compute gradient
    guidance_grad = torch.autograd.grad(
        outputs=measurement_loss,
        inputs=x_t,
        retain_graph=False,
    )[0]
    
    # Normalize gradient
    grad_norm = torch.norm(guidance_grad)
    if grad_norm > gradient_clip:
        guidance_grad = guidance_grad * gradient_clip / grad_norm
    
    return x_0_hat.detach(), guidance_grad.detach()


# ============== Sampling Algorithms ==============

def sample_ddpm_guided(
    model: torch.nn.Module,
    diffusion: EDMDiffusion,
    y_clipped: torch.Tensor,
    clip_level: float,
    num_steps: int = 50,
    guidance_weight: float = 1.0,
    stochasticity: float = 0.0,  # 0 = deterministic (DDIM), 1 = full stochastic (DDPM)
    device: torch.device = None,
) -> tuple:
    """
    Sample with reconstruction guidance (DDPM/DDIM style).
    
    Args:
        model: Trained denoiser
        diffusion: Diffusion parameters
        y_clipped: Clipped observation, shape (B, 2, T)
        clip_level: Clipping threshold
        num_steps: Number of sampling steps
        guidance_weight: Lambda - higher = stronger measurement consistency
        stochasticity: Noise injection amount (0=deterministic, 1=full)
        device: Device
        
    Returns:
        x_0: Reconstructed signal
        intermediates: List of intermediate samples (for visualization)
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    shape = y_clipped.shape
    
    # Get sigma schedule
    sigmas = diffusion.get_schedule(num_steps, device)
    
    # Initialize from noise (or from clipped observation)
    # Option 1: Pure noise
    # x = torch.randn(shape, device=device) * sigmas[0]
    
    # Option 2: Initialize from clipped + noise (warm start)
    x = y_clipped + torch.randn(shape, device=device) * sigmas[0]
    
    intermediates = []
    
    with torch.no_grad():
        for i in tqdm(range(num_steps), desc="Sampling"):
            sigma_t = sigmas[i].view(1, 1)  # Current
            sigma_next = sigmas[i + 1].view(1, 1)  # Next
            
            if sigma_t == 0:
                break
            
            # Reconstruction guidance step
            x_0_hat, guidance_grad = reconstruction_guidance_step(
                model, diffusion, x, sigma_t.expand(shape[0], 1),
                y_clipped, clip_level, guidance_weight
            )
            
            # Score from denoiser: score = (x_0_hat - x) / sigma^2
            score = (x_0_hat - x) / (sigma_t ** 2)
            
            # Apply guidance: score_guided = score - lambda * grad
            # Note: grad points towards higher loss, so we subtract
            score_guided = score - guidance_weight * guidance_grad / sigma_t
            
            # Euler step for probability flow ODE
            # dx = -sigma * score * dsigma
            dsigma = sigma_next - sigma_t
            x = x + dsigma * (-sigma_t * score_guided)
            
            # Optional stochasticity (Langevin correction)
            if stochasticity > 0 and sigma_next > 0:
                noise_scale = stochasticity * torch.sqrt(
                    sigma_t ** 2 - sigma_next ** 2
                )
                x = x + noise_scale * torch.randn_like(x)
            
            intermediates.append(x_0_hat.cpu())
    
    return x, intermediates


def sample_heun_guided(
    model: torch.nn.Module,
    diffusion: EDMDiffusion,
    y_clipped: torch.Tensor,
    clip_level: float,
    num_steps: int = 50,
    guidance_weight: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample with Heun's method (2nd order) and reconstruction guidance.
    
    More accurate than Euler but requires 2 function evaluations per step.
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    shape = y_clipped.shape
    sigmas = diffusion.get_schedule(num_steps, device)
    
    # Initialize
    x = y_clipped + torch.randn(shape, device=device) * sigmas[0]
    
    with torch.no_grad():
        for i in tqdm(range(num_steps), desc="Heun Sampling"):
            sigma_t = sigmas[i].view(1, 1)
            sigma_next = sigmas[i + 1].view(1, 1)
            
            if sigma_t == 0:
                break
            
            # First evaluation
            x_0_hat, guidance_grad = reconstruction_guidance_step(
                model, diffusion, x, sigma_t.expand(shape[0], 1),
                y_clipped, clip_level, guidance_weight
            )
            
            score = (x_0_hat - x) / (sigma_t ** 2)
            score_guided = score - guidance_weight * guidance_grad / sigma_t
            d = -sigma_t * score_guided
            
            if sigma_next == 0:
                # Last step: just use Euler
                x = x + (sigma_next - sigma_t) * d
            else:
                # Heun's method: second evaluation
                x_prime = x + (sigma_next - sigma_t) * d
                
                x_0_hat_prime, guidance_grad_prime = reconstruction_guidance_step(
                    model, diffusion, x_prime, sigma_next.expand(shape[0], 1),
                    y_clipped, clip_level, guidance_weight
                )
                
                score_prime = (x_0_hat_prime - x_prime) / (sigma_next ** 2)
                score_guided_prime = score_prime - guidance_weight * guidance_grad_prime / sigma_next
                d_prime = -sigma_next * score_guided_prime
                
                # Average
                x = x + (sigma_next - sigma_t) * 0.5 * (d + d_prime)
    
    return x


# ============== Main Inference ==============

def main():
    parser = argparse.ArgumentParser(description="OFDM Declipping Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--num_steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--guidance_weight", type=float, default=1.0, help="Guidance strength")
    parser.add_argument("--clip_sdr", type=float, default=5.0, help="Target clipping SDR (dB)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--output_dir", type=str, default="results/declipping")
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun"])
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() 
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    # Create model
    model = UNet1DComplexSimple(
        in_channels=2,
        base_channels=config["base_channels"],
        depth=config["depth"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Loaded model from step {checkpoint['step']}")
    
    # Diffusion
    diffusion = EDMDiffusion(sigma_data=1.0)
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results = []
    
    for sample_idx in range(args.num_samples):
        print(f"\n{'='*50}")
        print(f"Sample {sample_idx + 1}/{args.num_samples}")
        
        # Generate clean OFDM signal
        ofdm_config = OFDMConfig()
        signal_complex, metadata = generate_ofdm_signal_complex(
            ofdm_config, config["signal_length"], seed=sample_idx * 1000
        )
        x_clean = torch.from_numpy(complex_to_2channel(signal_complex)).unsqueeze(0).to(device)
        
        # Determine clip level for target SDR
        # Binary search to find clip level
        def find_clip_level(x, target_sdr, tol=0.5):
            low, high = 0.01, 3.0
            for _ in range(20):
                mid = (low + high) / 2
                clipped = clip_signal(x, mid)
                sdr = compute_clip_sdr(x, clipped)
                if abs(sdr - target_sdr) < tol:
                    return mid
                elif sdr > target_sdr:
                    high = mid
                else:
                    low = mid
            return mid
        
        clip_level = find_clip_level(x_clean, args.clip_sdr)
        
        # Clip the signal
        y_clipped = clip_signal(x_clean, clip_level)
        actual_sdr = compute_clip_sdr(x_clean, y_clipped)
        print(f"Clipping level: {clip_level:.4f}, SDR: {actual_sdr:.2f} dB")
        
        # Compute EVM before reconstruction
        y_clipped_np = y_clipped[0].cpu().numpy()
        clipped_symbols = demodulate_ofdm_complex(y_clipped_np, metadata)
        evm_before = compute_evm_complex(metadata['data_symbols'], clipped_symbols)
        print(f"EVM before declipping: {evm_before:.2f}%")
        
        # Run declipping
        if args.sampler == "euler":
            x_recon, intermediates = sample_ddpm_guided(
                model, diffusion, y_clipped, clip_level,
                num_steps=args.num_steps,
                guidance_weight=args.guidance_weight,
                device=device
            )
        else:
            x_recon = sample_heun_guided(
                model, diffusion, y_clipped, clip_level,
                num_steps=args.num_steps,
                guidance_weight=args.guidance_weight,
                device=device
            )
        
        # Compute EVM after reconstruction
        x_recon_np = x_recon[0].cpu().numpy()
        recon_symbols = demodulate_ofdm_complex(x_recon_np, metadata)
        evm_after = compute_evm_complex(metadata['data_symbols'], recon_symbols)
        print(f"EVM after declipping: {evm_after:.2f}%")
        
        # Also compute MSE
        mse_before = F.mse_loss(y_clipped, x_clean).item()
        mse_after = F.mse_loss(x_recon, x_clean).item()
        print(f"MSE: {mse_before:.6f} -> {mse_after:.6f}")
        
        results.append({
            'sample_idx': sample_idx,
            'clip_sdr': actual_sdr,
            'clip_level': clip_level,
            'evm_before': evm_before,
            'evm_after': evm_after,
            'mse_before': mse_before,
            'mse_after': mse_after,
            'modulation': metadata['modulation'],
            'fft_size': metadata['fft_size'],
        })
        
        # Save signals
        np.save(output_dir / f"sample_{sample_idx}_clean.npy", x_clean[0].cpu().numpy())
        np.save(output_dir / f"sample_{sample_idx}_clipped.npy", y_clipped[0].cpu().numpy())
        np.save(output_dir / f"sample_{sample_idx}_recon.npy", x_recon[0].cpu().numpy())
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    
    avg_evm_before = np.mean([r['evm_before'] for r in results])
    avg_evm_after = np.mean([r['evm_after'] for r in results])
    avg_improvement = avg_evm_before - avg_evm_after
    
    print(f"Average EVM before: {avg_evm_before:.2f}%")
    print(f"Average EVM after:  {avg_evm_after:.2f}%")
    print(f"Average improvement: {avg_improvement:.2f}%")
    
    # Save results
    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
