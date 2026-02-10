"""
OFDM Declipping with Diffusion + Reconstruction Guidance
=========================================================
FIXED version that prevents constellation collapse.

Key fixes:
1. Proper measurement loss with gradients for clipped samples
2. Guidance weight scheduling (weak early, strong late)
3. OFDM structure projection (pilots, nulls, guard bands)
4. Prior sanity checks

Usage:
    python ofdm_declip_fixed.py --checkpoint demo_results/model.pt
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from tqdm import tqdm


# ============== PART 1: MEASUREMENT LOSS WITH PROPER GRADIENTS ==============

@dataclass
class ClipMeasurementConfig:
    """Configuration for clipping measurement loss."""
    gamma_unclipped: float = 1.0      # Weight for unclipped sample MSE
    eta_magnitude: float = 0.5        # Weight for magnitude hinge (clipped)
    eta_phase: float = 0.5            # Weight for phase alignment (clipped)
    epsilon: float = 1e-8             # Numerical stability
    soft_hinge_temp: float = 0.1      # Temperature for soft hinge


def compute_measurement_loss_complex(
    x_hat: torch.Tensor,           # Predicted clean signal [B, 2, T] (I/Q channels)
    y_clipped: torch.Tensor,       # Observed clipped signal [B, 2, T]
    clip_threshold: float,         # Clipping threshold T
    config: ClipMeasurementConfig = None
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute measurement loss with proper gradients for both clipped and unclipped samples.
    
    For complex signal represented as 2 channels [I, Q]:
    - Unclipped samples (|y| < T): L_unclip = ||x̂ - y||²
    - Clipped samples (|y| = T):
        - Magnitude hinge: max(0, T - |x̂|)² encourages |x̂| ≥ T
        - Phase alignment: 1 - cos(angle(x̂, y)) encourages same direction
    
    Returns:
        total_loss: Combined measurement loss
        loss_dict: Individual loss components for logging
    """
    if config is None:
        config = ClipMeasurementConfig()
    
    B, C, T = x_hat.shape
    assert C == 2, "Expected 2 channels (I/Q)"
    
    # Compute magnitudes
    # x_hat: [B, 2, T] -> magnitude [B, T]
    x_hat_mag = torch.sqrt(x_hat[:, 0]**2 + x_hat[:, 1]**2 + config.epsilon)
    y_mag = torch.sqrt(y_clipped[:, 0]**2 + y_clipped[:, 1]**2 + config.epsilon)
    
    # Identify clipped vs unclipped samples
    # Use small tolerance for numerical stability
    tolerance = 0.01 * clip_threshold
    is_clipped = (y_mag >= clip_threshold - tolerance).float()  # [B, T]
    is_unclipped = 1.0 - is_clipped
    
    # ============ LOSS 1: Unclipped samples - exact match ============
    # These samples weren't affected by clipping, so x̂ should equal y exactly
    diff = x_hat - y_clipped  # [B, 2, T]
    mse_per_sample = (diff[:, 0]**2 + diff[:, 1]**2)  # [B, T]
    
    # Only count unclipped samples
    unclipped_count = is_unclipped.sum() + config.epsilon
    loss_unclipped = (mse_per_sample * is_unclipped).sum() / unclipped_count
    
    # ============ LOSS 2: Clipped samples - magnitude hinge ============
    # For clipped samples, we know |x_true| >= T (it was clipped down to T)
    # Use soft hinge: max(0, T - |x̂|)² with smooth approximation
    
    # Soft hinge using softplus for gradient smoothness
    magnitude_deficit = clip_threshold - x_hat_mag  # Positive if |x̂| < T
    # Soft hinge: softplus(deficit)² gives gradients everywhere
    hinge_loss = F.softplus(magnitude_deficit / config.soft_hinge_temp) * config.soft_hinge_temp
    hinge_loss = hinge_loss ** 2
    
    clipped_count = is_clipped.sum() + config.epsilon
    loss_magnitude = (hinge_loss * is_clipped).sum() / clipped_count
    
    # ============ LOSS 3: Clipped samples - phase/direction alignment ============
    # The clipped signal preserves the phase/direction of the original
    # Encourage x̂ to point in the same direction as y
    
    # Normalize to unit vectors
    x_hat_norm = x_hat / (x_hat_mag.unsqueeze(1) + config.epsilon)  # [B, 2, T]
    y_norm = y_clipped / (y_mag.unsqueeze(1) + config.epsilon)
    
    # Cosine similarity: dot product of unit vectors
    cos_sim = (x_hat_norm[:, 0] * y_norm[:, 0] + x_hat_norm[:, 1] * y_norm[:, 1])  # [B, T]
    
    # Loss: 1 - cos_sim (0 when aligned, 2 when opposite)
    phase_loss = 1.0 - cos_sim
    loss_phase = (phase_loss * is_clipped).sum() / clipped_count
    
    # ============ COMBINE ============
    total_loss = (
        config.gamma_unclipped * loss_unclipped +
        config.eta_magnitude * loss_magnitude +
        config.eta_phase * loss_phase
    )
    
    loss_dict = {
        'loss_unclipped': loss_unclipped.detach(),
        'loss_magnitude': loss_magnitude.detach(),
        'loss_phase': loss_phase.detach(),
        'pct_clipped': (is_clipped.sum() / (B * T)).detach(),
    }
    
    return total_loss, loss_dict


# ============== PART 2: GUIDANCE WEIGHT SCHEDULING ==============

def guidance_schedule(
    sigma: float,
    sigma_max: float = 50.0,
    sigma_min: float = 0.01,
    schedule_type: str = "cosine",
    lambda_max: float = 2.0,
    lambda_min: float = 0.0,
    warmup_ratio: float = 0.3,
) -> float:
    """
    Compute guidance weight λ(σ) that varies across diffusion steps.
    
    Key insight: At high noise (large σ), the denoiser output x̂₀ is unreliable,
    so guidance should be weak. Near the end (small σ), x̂₀ is accurate and
    guidance should be strong.
    
    Args:
        sigma: Current noise level
        sigma_max/min: Noise schedule bounds
        schedule_type: "cosine", "linear", "step", or "exponential"
        lambda_max: Maximum guidance weight (at low sigma)
        lambda_min: Minimum guidance weight (at high sigma)
        warmup_ratio: Fraction of schedule where guidance is near-zero
    
    Returns:
        lambda_t: Guidance weight at this step
    """
    # Normalize sigma to [0, 1] where 0 = sigma_max, 1 = sigma_min
    log_sigma = np.log(sigma)
    log_max = np.log(sigma_max)
    log_min = np.log(sigma_min)
    t = (log_max - log_sigma) / (log_max - log_min + 1e-8)
    t = np.clip(t, 0, 1)
    
    if schedule_type == "cosine":
        # Smooth cosine schedule
        if t < warmup_ratio:
            # Near-zero during warmup (high noise)
            return lambda_min
        else:
            # Cosine ramp from warmup to end
            t_adj = (t - warmup_ratio) / (1 - warmup_ratio)
            weight = 0.5 * (1 - np.cos(np.pi * t_adj))
            return lambda_min + weight * (lambda_max - lambda_min)
    
    elif schedule_type == "linear":
        if t < warmup_ratio:
            return lambda_min
        else:
            t_adj = (t - warmup_ratio) / (1 - warmup_ratio)
            return lambda_min + t_adj * (lambda_max - lambda_min)
    
    elif schedule_type == "step":
        # Step function: zero until threshold, then constant
        return lambda_max if t > warmup_ratio else lambda_min
    
    elif schedule_type == "exponential":
        # Exponential ramp
        if t < warmup_ratio:
            return lambda_min
        else:
            t_adj = (t - warmup_ratio) / (1 - warmup_ratio)
            return lambda_min + (np.exp(3 * t_adj) - 1) / (np.exp(3) - 1) * (lambda_max - lambda_min)
    
    else:
        raise ValueError(f"Unknown schedule: {schedule_type}")


# ============== PART 3: OFDM STRUCTURE PROJECTION ==============

@dataclass
class OFDMProjectionConfig:
    """Configuration for OFDM structure projection."""
    fft_size: int = 256
    cp_length: int = 32
    num_symbols: int = 14
    pilot_indices: Optional[np.ndarray] = None      # Indices of pilot subcarriers
    pilot_values: Optional[np.ndarray] = None       # Known pilot values
    null_indices: Optional[np.ndarray] = None       # DC, guard bands
    data_indices: Optional[np.ndarray] = None       # Data subcarrier indices


def ofdm_structure_projection(
    x: torch.Tensor,
    config: OFDMProjectionConfig,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Project signal onto valid OFDM structure.
    
    Steps:
    1. Remove CP, FFT to frequency domain
    2. Overwrite pilot subcarriers with known values
    3. Zero out null/guard subcarriers  
    4. IFFT back, add CP
    
    This helps EVM because:
    - Pilots are KNOWN → zero error contribution
    - Nulls should be ZERO → no spurious energy
    - Constrains solution space → tighter constellation
    
    Args:
        x: Time-domain signal [B, 2, T] (I/Q channels)
        config: OFDM parameters
    
    Returns:
        x_proj: Projected signal [B, 2, T]
    """
    B, C, T = x.shape
    fft_size = config.fft_size
    cp_len = config.cp_length
    symbol_len = fft_size + cp_len
    num_symbols = config.num_symbols
    original_device = x.device
    
    # Move to CPU for complex operations (MPS doesn't support torch.complex)
    x_cpu = x.cpu()
    
    # Convert to complex
    x_complex = torch.complex(x_cpu[:, 0], x_cpu[:, 1])  # [B, T]
    
    # Process each OFDM symbol
    projected_symbols = []
    
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_len
        end = start + symbol_len
        
        if end > T:
            break
        
        # Extract symbol (with CP)
        symbol_with_cp = x_complex[:, start:end]  # [B, symbol_len]
        
        # Remove CP
        symbol = symbol_with_cp[:, cp_len:]  # [B, fft_size]
        
        # FFT to frequency domain
        freq = torch.fft.fft(symbol, dim=-1)  # [B, fft_size]
        
        # Apply constraints
        if config.pilot_indices is not None and config.pilot_values is not None:
            # Overwrite pilots with known values
            pilot_vals = torch.tensor(config.pilot_values, dtype=freq.dtype, device=freq.device)
            freq[:, config.pilot_indices] = pilot_vals.unsqueeze(0).expand(B, -1)
        
        if config.null_indices is not None:
            # Zero out null subcarriers (DC, guards)
            freq[:, config.null_indices] = 0
        
        # IFFT back to time domain
        symbol_proj = torch.fft.ifft(freq, dim=-1)
        
        # Add CP back
        cp = symbol_proj[:, -cp_len:]
        symbol_with_cp_proj = torch.cat([cp, symbol_proj], dim=-1)
        
        projected_symbols.append(symbol_with_cp_proj)
    
    # Concatenate all symbols
    if len(projected_symbols) > 0:
        x_proj_complex = torch.cat(projected_symbols, dim=-1)
        
        # Pad or trim to original length
        if x_proj_complex.shape[-1] < T:
            padding = T - x_proj_complex.shape[-1]
            x_proj_complex = F.pad(x_proj_complex, (0, padding))
        else:
            x_proj_complex = x_proj_complex[:, :T]
        
        # Convert back to 2-channel
        x_proj = torch.stack([x_proj_complex.real, x_proj_complex.imag], dim=1)
    else:
        x_proj = x_cpu
    
    # Move back to original device
    return x_proj.to(original_device)


# ============== PART 4: COMPLETE INFERENCE ALGORITHM ==============

def declip_ofdm_fixed(
    model: torch.nn.Module,
    y_clipped: torch.Tensor,
    clip_threshold: float,
    ofdm_config: OFDMProjectionConfig,
    num_steps: int = 50,
    sigma_max: float = 50.0,
    sigma_min: float = 0.01,
    guidance_lambda_max: float = 2.0,
    guidance_schedule_type: str = "cosine",
    measurement_config: ClipMeasurementConfig = None,
    apply_ofdm_projection: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> torch.Tensor:
    """
    FIXED OFDM declipping with diffusion + reconstruction guidance.
    
    Algorithm:
    1. Initialize from clipped observation + noise
    2. For each diffusion step:
        a. Denoise to get x̂₀
        b. Compute measurement loss L_meas with proper gradients
        c. Get guidance weight λ(σ) from schedule  
        d. Update with guided score
        e. Apply OFDM structure projection
    3. Return reconstructed signal
    
    Args:
        model: Trained diffusion denoiser
        y_clipped: Clipped observation [B, 2, T]
        clip_threshold: Clipping level T
        ofdm_config: OFDM parameters for projection
        num_steps: Sampling steps
        sigma_max/min: Noise schedule
        guidance_lambda_max: Maximum guidance weight
        guidance_schedule_type: How λ varies with σ
        measurement_config: Weights for loss components
        apply_ofdm_projection: Whether to project onto OFDM structure
        device: Compute device
        verbose: Print progress
    
    Returns:
        x_recon: Reconstructed signal [B, 2, T]
    """
    model.eval()
    
    if measurement_config is None:
        measurement_config = ClipMeasurementConfig()
    
    B, C, T = y_clipped.shape
    
    # Noise schedule (geometric)
    sigmas = torch.exp(
        torch.linspace(np.log(sigma_max), np.log(sigma_min), num_steps + 1)
    ).to(device)
    sigmas[-1] = 0  # Final step is noiseless
    
    # Initialize: start from clipped observation + initial noise
    x = y_clipped + torch.randn_like(y_clipped) * sigmas[0]
    
    iterator = tqdm(range(num_steps), desc="Declipping", disable=not verbose)
    
    loss_history = []
    
    for i in iterator:
        sigma_t = sigmas[i].item()
        sigma_next = sigmas[i + 1].item()
        
        if sigma_t == 0:
            break
        
        # Get guidance weight for this step
        lambda_t = guidance_schedule(
            sigma_t, sigma_max, sigma_min,
            schedule_type=guidance_schedule_type,
            lambda_max=guidance_lambda_max,
            lambda_min=0.0,
            warmup_ratio=0.3,
        )
        
        # ============ STEP A: Denoise to get x̂₀ ============
        x_in = x.detach().requires_grad_(True)
        
        with torch.enable_grad():
            # Model predicts denoised signal
            sigma_batch = torch.full((B, 1), sigma_t, device=device)
            x_0_hat = model(x_in, sigma_batch)
            
            # ============ STEP B: Compute measurement loss ============
            loss_meas, loss_dict = compute_measurement_loss_complex(
                x_0_hat, y_clipped, clip_threshold, measurement_config
            )
            
            # ============ STEP C: Gradient for guidance ============
            grad = torch.autograd.grad(loss_meas, x_in)[0]
        
        x_0_hat = x_0_hat.detach()
        
        # Log for debugging
        loss_history.append({
            'step': i,
            'sigma': sigma_t,
            'lambda': lambda_t,
            **{k: v.item() for k, v in loss_dict.items()},
        })
        
        if verbose and i % 10 == 0:
            iterator.set_postfix({
                'σ': f'{sigma_t:.3f}',
                'λ': f'{lambda_t:.2f}',
                'L': f'{loss_meas.item():.4f}',
            })
        
        # ============ STEP D: Guided score update ============
        # Score from denoiser
        score = (x_0_hat - x) / (sigma_t ** 2 + 1e-8)
        
        # Gradient clipping for stability
        grad_norm = torch.norm(grad)
        if grad_norm > 1.0:
            grad = grad / grad_norm
        
        # Guided score
        score_guided = score - lambda_t * grad
        
        # Euler-Maruyama step
        dt = sigma_next - sigma_t  # Negative (decreasing sigma)
        x = x + dt * (-sigma_t * score_guided)
        
        # ============ STEP E: OFDM structure projection ============
        if apply_ofdm_projection and i % 5 == 0:  # Every 5 steps
            x = ofdm_structure_projection(x, ofdm_config, device)
    
    # Final projection
    if apply_ofdm_projection:
        x = ofdm_structure_projection(x, ofdm_config, device)
    
    return x, loss_history


# ============== PART 5: SANITY CHECKS FOR PRIOR TRAINING ==============

def sanity_check_prior(model, device="cpu"):
    """
    Sanity checks to verify the diffusion prior is properly trained.
    Run these BEFORE attempting declipping.
    """
    print("\n" + "="*60)
    print("SANITY CHECKS FOR DIFFUSION PRIOR")
    print("="*60)
    
    model.eval()
    passed = 0
    total = 3
    
    # ============ CHECK 1: Single-step denoising ============
    print("\n[1/3] Single-step denoising test...")
    
    # Create a simple test signal
    x_clean = torch.randn(1, 2, 1024).to(device)
    sigma_test = 0.1  # Low noise
    x_noisy = x_clean + sigma_test * torch.randn_like(x_clean)
    
    with torch.no_grad():
        sigma_batch = torch.full((1, 1), sigma_test, device=device)
        x_denoised = model(x_noisy, sigma_batch)
    
    mse_before = F.mse_loss(x_noisy, x_clean).item()
    mse_after = F.mse_loss(x_denoised, x_clean).item()
    
    if mse_after < mse_before:
        print(f"  ✓ PASS: MSE reduced from {mse_before:.4f} to {mse_after:.4f}")
        passed += 1
    else:
        print(f"  ✗ FAIL: MSE increased from {mse_before:.4f} to {mse_after:.4f}")
        print("    → Model may not be properly trained or sigma conditioning is wrong")
    
    # ============ CHECK 2: Sigma sensitivity ============
    print("\n[2/3] Sigma sensitivity test...")
    
    with torch.no_grad():
        sigma_low = torch.full((1, 1), 0.01, device=device)
        sigma_high = torch.full((1, 1), 10.0, device=device)
        
        out_low = model(x_noisy, sigma_low)
        out_high = model(x_noisy, sigma_high)
    
    diff = F.mse_loss(out_low, out_high).item()
    
    if diff > 0.01:
        print(f"  ✓ PASS: Model outputs differ for σ=0.01 vs σ=10 (diff={diff:.4f})")
        passed += 1
    else:
        print(f"  ✗ FAIL: Model outputs nearly identical (diff={diff:.4f})")
        print("    → Sigma conditioning may not be working")
    
    # ============ CHECK 3: Output statistics ============
    print("\n[3/3] Output statistics test...")
    
    # Generate from pure noise
    x_noise = torch.randn(1, 2, 1024).to(device) * 50
    with torch.no_grad():
        sigma_batch = torch.full((1, 1), 50.0, device=device)
        x_out = model(x_noise, sigma_batch)
    
    out_std = x_out.std().item()
    out_mean = x_out.mean().item()
    
    if 0.1 < out_std < 10 and abs(out_mean) < 1:
        print(f"  ✓ PASS: Output has reasonable stats (mean={out_mean:.3f}, std={out_std:.3f})")
        passed += 1
    else:
        print(f"  ✗ FAIL: Output has unusual stats (mean={out_mean:.3f}, std={out_std:.3f})")
        print("    → Check normalization during training")
    
    print(f"\n{'='*60}")
    print(f"RESULT: {passed}/{total} checks passed")
    if passed < total:
        print("\n⚠️  WARNING: Prior may not be properly trained!")
        print("Common issues:")
        print("  - Normalization mismatch (training vs inference)")
        print("  - Wrong prediction target (x0 vs epsilon vs v)")
        print("  - Sigma range mismatch")
        print("  - Insufficient training steps")
    print("="*60)
    
    return passed == total


# ============== PART 6: RECOMMENDED HYPERPARAMETERS ==============

def get_recommended_config():
    """Get recommended hyperparameters for OFDM declipping."""
    
    config = {
        # Diffusion sampling
        'num_steps': 50,                    # More steps = better quality
        'sigma_max': 50.0,                  # Match training range
        'sigma_min': 0.01,                  # Don't go too low
        
        # Guidance schedule
        'guidance_lambda_max': 2.0,         # Maximum guidance strength
        'guidance_schedule': 'cosine',      # Smooth schedule
        'warmup_ratio': 0.3,                # No guidance for first 30% of steps
        
        # Measurement loss weights
        'gamma_unclipped': 1.0,             # Unclipped samples: exact match
        'eta_magnitude': 0.5,               # Clipped: magnitude constraint
        'eta_phase': 0.5,                   # Clipped: phase alignment
        'soft_hinge_temp': 0.1,             # Softness of hinge
        
        # OFDM projection
        'apply_projection': True,           # Highly recommended
        'projection_frequency': 5,          # Every N steps
    }
    
    return config


# ============== MAIN DEMO ==============

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="demo_results/model.pt")
    parser.add_argument("--clip_level", type=float, default=0.7)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_max", type=float, default=2.0)
    parser.add_argument("--skip_sanity", action="store_true")
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Import model and data generation
    from demo_ofdm_full_pipeline import (
        SimpleUNet, OFDMConfig, generate_ofdm_signal,
        complex_to_2ch, ch2_to_complex, demodulate_ofdm, compute_evm,
        estimate_channel_gain, equalize_symbols
    )
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = SimpleUNet(in_ch=2, base_ch=32, depth=4, emb_dim=256).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Sanity checks
    if not args.skip_sanity:
        prior_ok = sanity_check_prior(model, device)
        if not prior_ok:
            print("\n⚠️  Prior may not be trained. Continuing anyway...")
    
    # Generate test signal
    print("\nGenerating test OFDM signal...")
    ofdm_cfg = OFDMConfig()
    signal, metadata = generate_ofdm_signal(ofdm_cfg, 4096, seed=42)
    
    # Apply clipping
    max_amp = np.max(np.abs(signal))
    signal_norm = signal / max_amp
    magnitude = np.abs(signal_norm)
    phase = np.angle(signal_norm)
    clipped_mag = np.minimum(magnitude, args.clip_level)
    clipped = clipped_mag * np.exp(1j * phase) * max_amp
    
    # Convert to tensor
    y_clipped = torch.from_numpy(complex_to_2ch(clipped)).unsqueeze(0).float().to(device)
    
    # Setup OFDM projection config
    proj_config = OFDMProjectionConfig(
        fft_size=metadata['fft_size'],
        cp_length=int(metadata['cp_ratio'] * metadata['fft_size']),
        num_symbols=metadata['num_symbols'],
        null_indices=np.array([0]),  # DC null
    )
    
    # Setup measurement config
    meas_config = ClipMeasurementConfig(
        gamma_unclipped=1.0,
        eta_magnitude=0.5,
        eta_phase=0.5,
    )
    
    # Run fixed declipping
    print("\nRunning FIXED declipping algorithm...")
    x_recon, loss_history = declip_ofdm_fixed(
        model=model,
        y_clipped=y_clipped,
        clip_threshold=args.clip_level,
        ofdm_config=proj_config,
        num_steps=args.num_steps,
        guidance_lambda_max=args.guidance_max,
        guidance_schedule_type="cosine",
        measurement_config=meas_config,
        apply_ofdm_projection=True,
        device=device,
        verbose=True,
    )
    
    # Convert back and evaluate
    reconstructed = ch2_to_complex(x_recon.squeeze(0).cpu().numpy())
    
    # Compute EVM with per-signal LS gain equalization (3GPP-style)
    ref_symbols = metadata['data_symbols'].flatten()
    orig_symbols_raw = demodulate_ofdm(signal, metadata).flatten()
    clip_symbols_raw = demodulate_ofdm(clipped, metadata).flatten()
    recon_symbols_raw = demodulate_ofdm(reconstructed, metadata).flatten()

    orig_symbols = equalize_symbols(orig_symbols_raw, estimate_channel_gain(ref_symbols, orig_symbols_raw))
    clip_symbols = equalize_symbols(clip_symbols_raw, estimate_channel_gain(ref_symbols, clip_symbols_raw))
    recon_symbols = equalize_symbols(recon_symbols_raw, estimate_channel_gain(ref_symbols, recon_symbols_raw))

    evm_orig = compute_evm(ref_symbols, orig_symbols)
    evm_clip = compute_evm(ref_symbols, clip_symbols)
    evm_recon = compute_evm(ref_symbols, recon_symbols)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"EVM Original:      {evm_orig:.2f}%")
    print(f"EVM Clipped:       {evm_clip:.2f}%")
    print(f"EVM Reconstructed: {evm_recon:.2f}%")
    print(f"Improvement:       {evm_clip - evm_recon:.2f}%")
    
    # Plot
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # Time domain
    axes[0, 0].plot(signal.real[:300], 'b-', alpha=0.7)
    axes[0, 0].set_title('Original')
    axes[0, 1].plot(clipped.real[:300], 'r-', alpha=0.7)
    axes[0, 1].set_title('Clipped')
    axes[0, 2].plot(reconstructed.real[:300], 'g-', alpha=0.7)
    axes[0, 2].set_title('Reconstructed (Fixed)')
    
    # Constellations
    axes[1, 0].scatter(orig_symbols.real, orig_symbols.imag, alpha=0.5, s=5)
    axes[1, 0].scatter(ref_symbols.real, ref_symbols.imag, c='red', marker='x', s=50, alpha=0.3)
    axes[1, 0].set_title(f'Original (EVM={evm_orig:.1f}%)')
    axes[1, 0].axis('equal')
    
    axes[1, 1].scatter(clip_symbols.real, clip_symbols.imag, alpha=0.5, s=5)
    axes[1, 1].scatter(ref_symbols.real, ref_symbols.imag, c='red', marker='x', s=50, alpha=0.3)
    axes[1, 1].set_title(f'Clipped (EVM={evm_clip:.1f}%)')
    axes[1, 1].axis('equal')
    
    axes[1, 2].scatter(recon_symbols.real, recon_symbols.imag, alpha=0.5, s=5)
    axes[1, 2].scatter(ref_symbols.real, ref_symbols.imag, c='red', marker='x', s=50, alpha=0.3)
    axes[1, 2].set_title(f'Reconstructed (EVM={evm_recon:.1f}%)')
    axes[1, 2].axis('equal')
    
    plt.tight_layout()
    plt.savefig('declip_fixed_result.png', dpi=150)
    plt.show()
    
    print(f"\nPlot saved to declip_fixed_result.png")
