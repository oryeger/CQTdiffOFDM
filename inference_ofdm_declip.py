#!/usr/bin/env python3
"""
Inference-only script for OFDM Declipping.

Does exactly what demo_ofdm_full_pipeline.py does in Step 3 (Test Declipping),
but loads a pre-trained model instead of training.

Usage:
    python inference_ofdm_declip.py --checkpoint demo_results/model.pt
    python inference_ofdm_declip.py --checkpoint demo_results/model.pt --clip_level 1.5 --num_samples 5
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

# Import everything from the demo to ensure identical behavior
from demo_ofdm_full_pipeline import (
    OFDMConfig,
    generate_ofdm_signal,
    complex_to_2ch,
    ch2_to_complex,
    demodulate_ofdm,
    compute_evm,
    clip_signal,
    get_clipping_mask,
    plot_signal_comparison,
    sample_conditional_ddim,
    sample_with_guidance,
    ConditionalDiffusion,
    SimpleDiffusion,
    SimpleUNet,
)

# Import conditional model
try:
    from src.models.unet_ofdm_declip import ConditionalUNet1DSimple
    HAS_CONDITIONAL = True
except ImportError:
    HAS_CONDITIONAL = False


def main():
    parser = argparse.ArgumentParser(description="OFDM Declipping Inference (same as demo Step 3)")
    parser.add_argument("--checkpoint", type=str, default="demo_results/model.pt",
                        help="Path to trained model checkpoint")
    parser.add_argument("--clip_level", type=float, default=1.5,
                        help="Clip level as multiple of signal std (higher=less clipping)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of test samples")
    parser.add_argument("--sampling_steps", type=int, default=50,
                        help="Number of diffusion sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (conditional model)")
    parser.add_argument("--guidance_weight", type=float, default=1.0,
                        help="Guidance weight (simple model)")
    parser.add_argument("--signal_length", type=int, default=4096,
                        help="Signal length in samples")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=1000,
                        help="Base random seed for test signals")
    args = parser.parse_args()
    
    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        print("Train a model first with:")
        print("  python demo_ofdm_full_pipeline.py --model_type conditional --train_steps 5000")
        return
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Determine model type
    model_type = checkpoint.get('model_type', 'simple')
    config = checkpoint.get('config', {})
    
    print(f"Model type: {model_type}")
    
    # Create model based on type
    if model_type == 'conditional':
        if not HAS_CONDITIONAL:
            print("Error: Conditional model not available")
            return
        
        model = ConditionalUNet1DSimple(
            in_channels=2,
            out_channels=2,
            base_channels=config.get('base_channels', 64),
            depth=config.get('depth', 4),
            embed_dim=config.get('embed_dim', 512),
            cond_drop_prob=0.0,  # No dropout during inference
        ).to(device)
        
        diffusion = ConditionalDiffusion()
    else:
        model = SimpleUNet(
            in_ch=2,
            base_ch=config.get('base_ch', 32),
            depth=config.get('depth', 4),
        ).to(device)
        
        diffusion = SimpleDiffusion()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params/1e6:.2f}M")
    
    # OFDM config (same as demo)
    ofdm_config = OFDMConfig()
    
    # ========== TEST DECLIPPING (Same as demo Step 3) ==========
    print("\n" + "="*60)
    print("OFDM DECLIPPING INFERENCE")
    print("="*60)
    
    all_results = []
    
    for test_idx in range(args.num_samples):
        print(f"\n--- Test Sample {test_idx + 1}/{args.num_samples} ---")
        
        # Generate test signal (same as demo)
        test_signal, test_metadata = generate_ofdm_signal(
            ofdm_config, args.signal_length, seed=args.seed + test_idx
        )
        
        print(f"  Modulation: {test_metadata['modulation']}")
        print(f"  FFT size: {test_metadata['fft_size']}")
        
        # Convert to tensor
        x_clean = torch.from_numpy(complex_to_2ch(test_signal)).unsqueeze(0).to(device)
        
        # Apply clipping (same as demo)
        clip_level = args.clip_level * torch.std(x_clean).item()
        x_clipped = clip_signal(x_clean, clip_level)
        
        # Compute clipping SDR
        distortion = x_clean - x_clipped
        sdr = 10 * torch.log10(torch.mean(x_clean**2) / (torch.mean(distortion**2) + 1e-10))
        print(f"  Clipping SDR: {sdr.item():.2f} dB")
        
        # Get OFDM parameters
        fft_size = test_metadata['fft_size']
        cp_length = int(test_metadata['cp_ratio'] * fft_size)
        num_symbols = test_metadata['num_symbols']
        modulation = test_metadata['modulation']
        
        # Run declipping (same as demo)
        if model_type == "conditional":
            print("  Running declipping (CONDITIONAL model + DDIM)...")
            
            # Create mask based on MAGNITUDE
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
            print("  Running declipping (SIMPLE model + GUIDANCE)...")
            x_recon = sample_with_guidance(
                model, diffusion, x_clipped, clip_level,
                num_steps=args.sampling_steps,
                guidance_weight=args.guidance_weight,
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
        original_np = ch2_to_complex(x_clean[0].cpu().numpy())
        clipped_np = ch2_to_complex(x_clipped[0].cpu().numpy())
        recon_np = ch2_to_complex(x_recon[0].cpu().numpy())
        
        # Demodulate
        original_symbols = demodulate_ofdm(original_np, test_metadata).flatten()
        clipped_symbols = demodulate_ofdm(clipped_np, test_metadata).flatten()
        recon_symbols = demodulate_ofdm(recon_np, test_metadata).flatten()
        
        # Plot comparison (same as demo)
        evm_orig, evm_clip, evm_recon = plot_signal_comparison(
            original_np, clipped_np, recon_np,
            original_symbols, clipped_symbols, recon_symbols,
            test_metadata,
            save_path=output_dir / f"test_{test_idx+1}.png"
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
    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nAll results saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
