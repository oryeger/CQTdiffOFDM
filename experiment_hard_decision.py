"""
Experiment: Hard-decision declipping baseline.

Pipeline:
  clipped signal (time) → demodulate (FFT) → snap to nearest constellation → remodulate (IFFT)

No ML, no diffusion. Just signal processing.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from demo_ofdm_full_pipeline import (
    OFDMConfig, generate_ofdm_signal, demodulate_ofdm,
    compute_evm, estimate_channel_gain, equalize_symbols,
    generate_qam_constellation,
)


def clip_signal_np(signal, clip_level):
    """Magnitude-based clipping of complex signal."""
    magnitude = np.abs(signal)
    scale = np.minimum(1.0, clip_level / (magnitude + 1e-10))
    return signal * scale


def remodulate_ofdm(symbols, metadata):
    """
    Remodulate: frequency-domain symbols → time-domain OFDM signal.
    Inverse of demodulate_ofdm.
    """
    fft_size = metadata['fft_size']
    cp_length = metadata['cp_length']
    num_symbols = metadata['num_symbols']
    data_indices = metadata['data_indices']

    all_symbols = []
    for sym_idx in range(num_symbols):
        # Build frequency-domain symbol
        freq = np.zeros(fft_size, dtype=complex)
        freq[data_indices] = symbols[sym_idx]

        # IFFT (match generation: ifft * sqrt(N))
        time_symbol = np.fft.ifft(freq) * np.sqrt(fft_size)

        # Add cyclic prefix
        if cp_length > 0:
            cp = time_symbol[-cp_length:]
            time_symbol_cp = np.concatenate([cp, time_symbol])
        else:
            time_symbol_cp = time_symbol

        all_symbols.append(time_symbol_cp)

    signal = np.concatenate(all_symbols)
    return signal


def snap_to_constellation(symbols, constellation):
    """Snap each symbol to nearest constellation point."""
    flat = symbols.flatten()
    distances = np.abs(flat[:, None] - constellation[None, :])
    nearest_idx = np.argmin(distances, axis=1)
    snapped = constellation[nearest_idx]
    return snapped.reshape(symbols.shape)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_length", type=int, default=4096)
    parser.add_argument("--clip_level", type=float, default=1.5,
                        help="Clip level as multiple of signal std")
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="demo_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OFDMConfig()
    constellation = generate_qam_constellation("QPSK")

    print("=" * 60)
    print("Hard-Decision Declipping Baseline")
    print("=" * 60)
    print(f"  Clip level: {args.clip_level} x std")
    print(f"  Modulation: QPSK")
    print()

    all_evm_orig = []
    all_evm_clip = []
    all_evm_recon = []
    all_ser = []

    fig, axes = plt.subplots(args.num_test, 4, figsize=(18, 4 * args.num_test))
    if args.num_test == 1:
        axes = axes[np.newaxis, :]

    for i in range(args.num_test):
        # 1. Generate clean signal
        signal, metadata = generate_ofdm_signal(config, args.signal_length, seed=1000 + i)

        # 2. Clip
        clip_level = args.clip_level * np.std(signal)
        clipped = clip_signal_np(signal, clip_level)

        # 3. Demodulate both
        ref_symbols = metadata['data_symbols']  # True transmitted symbols
        clip_symbols = demodulate_ofdm(clipped, metadata)

        # 4. Equalize clipped symbols (remove gain bias)
        ref_flat = ref_symbols.flatten()
        clip_flat = clip_symbols.flatten()
        gain_clip = estimate_channel_gain(ref_flat, clip_flat)
        clip_eq = equalize_symbols(clip_flat, gain_clip)

        # 5. Hard decision: snap equalized symbols to nearest constellation
        snapped = snap_to_constellation(clip_eq, constellation)

        # 6. Compute symbol error rate
        # Map ref to nearest constellation too (should be exact)
        ref_snapped = snap_to_constellation(ref_flat, constellation)
        num_errors = np.sum(np.abs(snapped - ref_snapped) > 0.01)
        ser = num_errors / len(ref_flat)
        all_ser.append(ser)

        # 7. Remodulate from snapped symbols (at original scale)
        snapped_2d = snapped.reshape(ref_symbols.shape)
        recon_signal = remodulate_ofdm(snapped_2d, metadata)

        # Pad/truncate to match
        if len(recon_signal) < args.signal_length:
            recon_signal = np.pad(recon_signal, (0, args.signal_length - len(recon_signal)))
        else:
            recon_signal = recon_signal[:args.signal_length]

        # Match scale to original
        recon_signal = recon_signal / np.std(recon_signal) * np.std(signal)

        # 8. Demodulate reconstructed and compute EVM
        recon_symbols = demodulate_ofdm(recon_signal, metadata)
        recon_flat = recon_symbols.flatten()
        gain_recon = estimate_channel_gain(ref_flat, recon_flat)
        recon_eq = equalize_symbols(recon_flat, gain_recon)

        evm_orig = compute_evm(ref_flat, ref_flat)  # Should be ~0
        evm_clip = compute_evm(ref_flat, clip_eq)
        evm_recon = compute_evm(ref_flat, recon_eq)

        all_evm_orig.append(evm_orig)
        all_evm_clip.append(evm_clip)
        all_evm_recon.append(evm_recon)

        print(f"  Signal {i+1}: EVM clipped={evm_clip:.2f}%  →  recon={evm_recon:.2f}%  "
              f"(improvement: {evm_clip - evm_recon:.2f}%)  SER={ser:.4f}")

        # Plot constellations
        axes[i, 0].scatter(clip_eq.real, clip_eq.imag, alpha=0.4, s=8, c='red')
        axes[i, 0].scatter(constellation.real, constellation.imag,
                           s=100, c='black', marker='x', linewidths=2, zorder=5)
        axes[i, 0].set_title(f'Clipped (EVM: {evm_clip:.1f}%)')
        axes[i, 0].axis('equal')
        axes[i, 0].set_xlim(-1.5, 1.5)
        axes[i, 0].set_ylim(-1.5, 1.5)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].scatter(snapped.real, snapped.imag, alpha=0.4, s=8, c='blue')
        axes[i, 1].scatter(constellation.real, constellation.imag,
                           s=100, c='black', marker='x', linewidths=2, zorder=5)
        axes[i, 1].set_title(f'Hard Decision (EVM: {evm_recon:.1f}%)')
        axes[i, 1].axis('equal')
        axes[i, 1].set_xlim(-1.5, 1.5)
        axes[i, 1].set_ylim(-1.5, 1.5)
        axes[i, 1].grid(True, alpha=0.3)

        # Time-domain: clipped vs original
        t = np.arange(300)
        axes[i, 2].plot(t, signal.real[:300], 'g-', alpha=0.5, label='Original')
        axes[i, 2].plot(t, clipped.real[:300], 'r-', alpha=0.7, label='Clipped')
        axes[i, 2].axhline(y=clip_level, color='r', linestyle='--', alpha=0.3)
        axes[i, 2].axhline(y=-clip_level, color='r', linestyle='--', alpha=0.3)
        axes[i, 2].set_title(f'Clipped vs Original')
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].grid(True, alpha=0.3)

        # Time-domain: reconstructed vs original
        axes[i, 3].plot(t, signal.real[:300], 'g-', alpha=0.5, label='Original')
        axes[i, 3].plot(t, recon_signal.real[:300], 'b--', alpha=0.7, label='Reconstructed')
        axes[i, 3].set_title(f'Reconstructed vs Original')
        axes[i, 3].legend(fontsize=8)
        axes[i, 3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "hard_decision_baseline.png", dpi=150)
    plt.show()

    # Summary
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Avg EVM clipped:       {np.mean(all_evm_clip):.2f}%")
    print(f"  Avg EVM reconstructed: {np.mean(all_evm_recon):.2f}%")
    print(f"  Avg improvement:       {np.mean(all_evm_clip) - np.mean(all_evm_recon):.2f}%")
    print(f"  Avg symbol error rate: {np.mean(all_ser):.4f}")
    print()
    if np.mean(all_ser) < 0.01:
        print("  → Hard decision recovers nearly all symbols correctly!")
        print("    For QPSK at this clipping level, ML may not be needed.")
    else:
        print(f"  → {np.mean(all_ser)*100:.1f}% symbol errors. Room for ML improvement.")


if __name__ == "__main__":
    main()
