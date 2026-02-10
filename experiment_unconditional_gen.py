"""
Experiment: Unconditional OFDM generation.

Train a diffusion model on clean OFDM signals, then generate from pure noise.
Measure EVM to check if the model has learned the OFDM prior.

No clipping, no guidance, no conditioning - just pure generation.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# Reuse OFDM generation and model from demo
from demo_ofdm_full_pipeline import (
    OFDMConfig, OFDMDataset,
    SimpleUNet, SimpleDiffusion,
    generate_ofdm_signal, demodulate_ofdm,
    complex_to_2ch, ch2_to_complex,
    compute_evm, estimate_channel_gain, equalize_symbols,
    generate_qam_constellation,
)


def sample_unconditional(model, diffusion, shape, num_steps=50, device="cpu"):
    """Pure unconditional sampling - just denoise from noise."""
    model.eval()
    sigmas = diffusion.get_schedule(num_steps, device)

    # Start from pure noise
    x = torch.randn(shape, device=device) * sigmas[0]

    for i in tqdm(range(num_steps), desc="Sampling"):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]

        if sigma_t == 0:
            break

        sigma_batch = sigma_t.view(1, 1).expand(shape[0], 1)

        with torch.no_grad():
            c_skip, c_out, c_in = diffusion.get_scalings(sigma_batch)
            model_input = c_in.unsqueeze(-1) * x
            model_output = model(model_input, sigma_batch)
            x_0_hat = c_skip.unsqueeze(-1) * x + c_out.unsqueeze(-1) * model_output

        # Euler step
        d = (x - x_0_hat) / sigma_t
        x = x + (sigma_next - sigma_t) * d

    return x


def compute_evm_nearest(symbols, constellation):
    """EVM relative to nearest constellation point (for generated signals without reference)."""
    symbols_flat = symbols.flatten()
    # Find nearest constellation point for each symbol
    distances = np.abs(symbols_flat[:, None] - constellation[None, :])
    nearest_idx = np.argmin(distances, axis=1)
    nearest_points = constellation[nearest_idx]
    # EVM
    error_power = np.mean(np.abs(symbols_flat - nearest_points) ** 2)
    ref_power = np.mean(np.abs(nearest_points) ** 2)
    return np.sqrt(error_power / (ref_power + 1e-10)) * 100


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_length", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of signals to generate")
    parser.add_argument("--output_dir", type=str, default="demo_results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OFDMConfig()

    # ===== 1. Train unconditional model =====
    print("=" * 60)
    print("Training unconditional diffusion model on clean OFDM signals")
    print("=" * 60)

    dataset = OFDMDataset(signal_length=args.signal_length, config=config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size)
    data_iter = iter(dataloader)

    model = SimpleUNet(in_ch=2, base_ch=args.base_ch, depth=4).to(device)
    diffusion = SimpleDiffusion()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params/1e6:.2f}M parameters, base_ch={args.base_ch}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Warmup + cosine schedule
    warmup_steps = min(200, args.train_steps // 10)

    losses = []
    pbar = tqdm(range(args.train_steps), desc="Training")
    for step in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # LR schedule
        if step < warmup_steps:
            lr = args.lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, args.train_steps - warmup_steps)
            lr = args.lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Standard diffusion training step (no tricks, no loss scaling)
        model.train()
        optimizer.zero_grad()

        x_0 = batch.to(device)
        B = x_0.shape[0]
        sigma = diffusion.sample_sigma(B, device)
        noise = torch.randn_like(x_0)
        x_noisy = x_0 + sigma.unsqueeze(-1) * noise

        c_skip, c_out, c_in = diffusion.get_scalings(sigma)
        model_input = c_in.unsqueeze(-1) * x_noisy
        model_output = model(model_input, sigma)
        x_pred = c_skip.unsqueeze(-1) * x_noisy + c_out.unsqueeze(-1) * model_output

        loss = F.mse_loss(x_pred, x_0)

        if torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 200 == 0:
            avg = np.mean(losses[-200:]) if losses else 0
            pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr:.1e}")

    print(f"Final loss: {np.mean(losses[-200:]):.6f}")

    # ===== 2. Generate reference: real OFDM signal EVM =====
    print("\n" + "=" * 60)
    print("Reference: Real OFDM signal EVM")
    print("=" * 60)

    constellation = generate_qam_constellation("QPSK")

    ref_signal, ref_meta = generate_ofdm_signal(config, args.signal_length, seed=42)
    ref_symbols = demodulate_ofdm(ref_signal, ref_meta)
    ref_evm = compute_evm_nearest(ref_symbols, constellation)
    print(f"  Real OFDM signal EVM (to nearest constellation): {ref_evm:.2f}%")

    # ===== 3. Generate signals from pure noise =====
    print("\n" + "=" * 60)
    print(f"Generating {args.num_samples} signals from pure noise")
    print("=" * 60)

    shape = (args.num_samples, 2, args.signal_length)
    generated = sample_unconditional(
        model, diffusion, shape,
        num_steps=args.sampling_steps, device=device
    )

    # ===== 4. Analyze generated signals =====
    # Use the same OFDM params as the training data for demodulation
    _, meta_template = generate_ofdm_signal(config, args.signal_length, seed=0)

    fig, axes = plt.subplots(args.num_samples, 3, figsize=(15, 4 * args.num_samples))
    if args.num_samples == 1:
        axes = axes[np.newaxis, :]

    gen_evms = []

    for i in range(args.num_samples):
        sig_np = ch2_to_complex(generated[i].cpu().numpy())

        # Demodulate using template OFDM params
        symbols = demodulate_ofdm(sig_np, meta_template)

        # Equalize: remove any overall gain/phase offset using nearest-point reference
        symbols_flat = symbols.flatten()
        nearest_idx = np.argmin(
            np.abs(symbols_flat[:, None] - constellation[None, :]), axis=1
        )
        nearest_ref = constellation[nearest_idx]
        gain = estimate_channel_gain(nearest_ref, symbols_flat)
        symbols_eq = equalize_symbols(symbols_flat, gain)

        evm = compute_evm_nearest(symbols_eq, constellation)
        gen_evms.append(evm)

        # Plot time domain
        axes[i, 0].plot(sig_np.real[:500], 'b-', alpha=0.7, label='I')
        axes[i, 0].plot(sig_np.imag[:500], 'r-', alpha=0.7, label='Q')
        axes[i, 0].set_title(f'Generated Signal {i+1}')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)

        # Plot constellation
        axes[i, 1].scatter(symbols_eq.real, symbols_eq.imag, alpha=0.4, s=8, c='blue')
        axes[i, 1].scatter(constellation.real, constellation.imag,
                           s=100, c='red', marker='x', linewidths=2, zorder=5)
        axes[i, 1].set_title(f'Constellation (EVM: {evm:.1f}%)')
        axes[i, 1].axis('equal')
        axes[i, 1].grid(True, alpha=0.3)
        axes[i, 1].set_xlim(-2, 2)
        axes[i, 1].set_ylim(-2, 2)

        # Plot magnitude histogram vs real OFDM
        gen_mag = np.abs(sig_np)
        ref_mag = np.abs(ref_signal)
        axes[i, 2].hist(ref_mag, bins=50, alpha=0.5, density=True, label='Real OFDM', color='green')
        axes[i, 2].hist(gen_mag, bins=50, alpha=0.5, density=True, label='Generated', color='blue')
        axes[i, 2].set_title('Magnitude Distribution')
        axes[i, 2].legend()
        axes[i, 2].grid(True, alpha=0.3)

        print(f"  Signal {i+1}: EVM = {evm:.2f}%")

    plt.tight_layout()
    plt.savefig(output_dir / "unconditional_generation.png", dpi=150)
    plt.show()

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Real OFDM EVM (nearest constellation): {ref_evm:.2f}%")
    print(f"  Generated signals EVM: {np.mean(gen_evms):.2f}% (std: {np.std(gen_evms):.2f}%)")
    print(f"  Train steps: {args.train_steps}, base_ch: {args.base_ch}")

    if np.mean(gen_evms) < 20:
        print("\n  -> Model HAS learned OFDM structure! Prior is good.")
        print("     Issue is likely in the conditioning/guidance for declipping.")
    elif np.mean(gen_evms) < 50:
        print("\n  -> Model has PARTIALLY learned OFDM structure.")
        print("     Try more training steps or larger model.")
    else:
        print("\n  -> Model has NOT learned OFDM structure.")
        print("     Need more capacity (base_ch) or more training steps.")


if __name__ == "__main__":
    main()
