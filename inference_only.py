"""
Inference-only script: Load trained model and declip OFDM samples.
No training - just reconstruction.

Usage:
    python inference_only.py --checkpoint demo_results/model.pt
    python inference_only.py --checkpoint demo_results/model.pt --clip_level 0.5
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import from demo script
from demo_ofdm_full_pipeline import (
    OFDMConfig,
    generate_ofdm_signal,
    complex_to_2ch,
    ch2_to_complex,
    demodulate_ofdm,
    compute_evm,
    estimate_channel_gain,
    equalize_symbols,
    SimpleUNet,
    SimpleDiffusion,
    sample_with_guidance,
    clip_signal,
    plot_signal_comparison,
)


def load_model(checkpoint_path: str, device: str = "cpu"):
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    # Create model architecture (must match training)
    model = SimpleUNet(in_ch=2, base_ch=32, depth=4, emb_dim=256).to(device)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"  Loaded from step {checkpoint.get('step', 'unknown')}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    return model


def generate_test_signal(config: OFDMConfig = None, length: int = 4096, seed: int = None):
    """Generate a test OFDM signal."""
    if config is None:
        config = OFDMConfig()
    if seed is None:
        seed = np.random.randint(0, 2**31)
    
    signal, metadata = generate_ofdm_signal(config, length, seed)
    return signal, metadata


def apply_clipping(signal: np.ndarray, clip_level: float = 0.7):
    """Apply hard clipping to signal."""
    # Normalize
    max_amp = np.max(np.abs(signal))
    normalized = signal / max_amp
    
    # Clip magnitude
    magnitude = np.abs(normalized)
    phase = np.angle(normalized)
    clipped_mag = np.minimum(magnitude, clip_level)
    clipped = clipped_mag * np.exp(1j * phase)
    
    # Restore scale
    return clipped * max_amp


def run_inference(
    model,
    clipped_signal: np.ndarray,
    original_signal: np.ndarray,
    clip_level: float,
    device: str,
    num_steps: int = 30,
    guidance_weight: float = 1.0,
    metadata: dict = None,
):
    """Run declipping inference with FIXED + CONSTELLATION guidance."""
    diffusion = SimpleDiffusion()
    
    # Convert to 2-channel tensor
    clipped_2ch = complex_to_2ch(clipped_signal)
    clipped_tensor = torch.from_numpy(clipped_2ch).unsqueeze(0).float().to(device)
    
    # Get OFDM parameters for projection
    if metadata is not None:
        fft_size = metadata['fft_size']
        cp_length = int(metadata['cp_ratio'] * fft_size)
        num_symbols = metadata['num_symbols']
        modulation = metadata.get('modulation', 'QPSK')
    else:
        fft_size = 256
        cp_length = 32
        num_symbols = 14
        modulation = 'QPSK'
    
    # Run declipping with FIXED + CONSTELLATION guidance
    reconstructed_tensor = sample_with_guidance(
        model, diffusion, clipped_tensor, 
        clip_level=clip_level,
        guidance_weight=guidance_weight,
        num_steps=num_steps,
        device=device,
        use_fixed_loss=True,
        apply_ofdm_proj=True,
        ofdm_fft_size=fft_size,
        ofdm_cp_length=cp_length,
        ofdm_num_symbols=num_symbols,
        modulation=modulation,
        use_constellation_loss=True,
        lambda_constellation=1.0,
    )
    
    # Convert back to complex
    reconstructed_2ch = reconstructed_tensor.squeeze(0).cpu().numpy()
    reconstructed = ch2_to_complex(reconstructed_2ch)
    
    return reconstructed


def main():
    parser = argparse.ArgumentParser(description="OFDM Declipping Inference")
    parser.add_argument("--checkpoint", type=str, default="demo_results/model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--clip_level", type=float, default=0.7,
                        help="Clipping level (0-1, lower = more clipping)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of test samples to process")
    parser.add_argument("--sampling_steps", type=int, default=30,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--guidance_weight", type=float, default=1.0,
                        help="Reconstruction guidance weight")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Process samples
    results = []
    
    for i in range(args.num_samples):
        print(f"\n{'='*60}")
        print(f"Sample {i+1}/{args.num_samples}")
        print(f"{'='*60}")
        
        # Generate test signal
        signal, metadata = generate_test_signal(seed=args.seed + i if args.seed else None)
        print(f"Generated signal: {len(signal)} samples, {metadata['modulation']}")
        
        # Apply clipping
        clipped = apply_clipping(signal, args.clip_level)
        print(f"Applied clipping at level {args.clip_level}")
        
        # Run inference with FIXED algorithm
        print(f"Running FIXED declipping ({args.sampling_steps} steps, guidance={args.guidance_weight})...")
        reconstructed = run_inference(
            model, clipped, signal, args.clip_level, device,
            num_steps=args.sampling_steps,
            guidance_weight=args.guidance_weight,
            metadata=metadata,
        )
        
        # Demodulate and compute EVM with bias removal
        ref_symbols = metadata['data_symbols'].flatten()
        orig_symbols_raw = demodulate_ofdm(signal, metadata).flatten()
        clip_symbols_raw = demodulate_ofdm(clipped, metadata).flatten()
        recon_symbols_raw = demodulate_ofdm(reconstructed, metadata).flatten()

        # Estimate and remove bias from time-domain normalization
        gain = estimate_channel_gain(ref_symbols, orig_symbols_raw)
        orig_symbols = equalize_symbols(orig_symbols_raw, gain)
        clip_symbols = equalize_symbols(clip_symbols_raw, gain)
        recon_symbols = equalize_symbols(recon_symbols_raw, gain)

        evm_orig = compute_evm(ref_symbols, orig_symbols)
        evm_clip = compute_evm(ref_symbols, clip_symbols)
        evm_recon = compute_evm(ref_symbols, recon_symbols)
        
        print(f"\nResults:")
        print(f"  EVM Original:      {evm_orig:.2f}%")
        print(f"  EVM Clipped:       {evm_clip:.2f}%")
        print(f"  EVM Reconstructed: {evm_recon:.2f}%")
        print(f"  Improvement:       {evm_clip - evm_recon:.2f}%")
        
        results.append({
            'evm_orig': evm_orig,
            'evm_clip': evm_clip,
            'evm_recon': evm_recon,
        })
        
        # Plot
        save_path = output_dir / f"sample_{i+1}.png"
        plot_signal_comparison(
            signal, clipped, reconstructed,
            orig_symbols, clip_symbols, recon_symbols,
            metadata, save_path=str(save_path)
        )
    
    # Summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        avg_clip = np.mean([r['evm_clip'] for r in results])
        avg_recon = np.mean([r['evm_recon'] for r in results])
        print(f"Average EVM Clipped:       {avg_clip:.2f}%")
        print(f"Average EVM Reconstructed: {avg_recon:.2f}%")
        print(f"Average Improvement:       {avg_clip - avg_recon:.2f}%")


if __name__ == "__main__":
    main()
