"""
Experiment: Frequency-domain diffusion for OFDM declipping.

Pipeline:
  clipped signal (time) → demodulate → diffusion denoises symbols → remodulate (time)

The diffusion model operates on frequency-domain symbols where the structure
(constellation points) is explicit and easy to learn.

FIXES (vs your previous version):
1) Equalization consistency:
   - Training equalizes BOTH clean and distorted using the gain estimated from CLEAN.
   - Testing must do the SAME (do NOT equalize clipped using a gain estimated from the clipped symbols),
     otherwise the model sees a shifted/scaled distribution and produces weird geometry.

2) Final hard decision (snapping) to constellation:
   - For 16QAM/64QAM, the posterior is multi-modal; diffusion often outputs averages (lines/frames).
   - Snapping each subcarrier symbol to the nearest constellation point fixes the “box/lines” artifact.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from demo_ofdm_full_pipeline import (
    OFDMConfig, generate_ofdm_signal, demodulate_ofdm,
    compute_evm, estimate_channel_gain,
    generate_qam_constellation,
)
from experiment_hard_decision import clip_signal_np, remodulate_ofdm


# ============== Helpers ==============

def snap_symbols_to_constellation(symbols_2d: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """
    Hard-decision snap complex symbols to nearest constellation points.

    Args:
        symbols_2d: complex array, shape (num_sym, N_sub)
        constellation: complex array, shape (M,)
    Returns:
        snapped: complex array, same shape as symbols_2d
    """
    # distances: (num_sym, N_sub, M)
    d2 = np.abs(symbols_2d[..., None] - constellation[None, None, :]) ** 2
    idx = np.argmin(d2, axis=-1)  # (num_sym, N_sub)
    return constellation[idx]


def complex_to_2ch_np(symbols: np.ndarray) -> np.ndarray:
    """(num_sym, N_sub) complex → (num_sym, 2, N_sub) float32"""
    return np.stack([symbols.real, symbols.imag], axis=1).astype(np.float32)


def ch2_to_complex_symbols(tensor: torch.Tensor) -> np.ndarray:
    """(B, 2, N_sub) tensor → (B, N_sub) complex numpy"""
    t = tensor.detach().cpu().numpy()
    return t[:, 0] + 1j * t[:, 1]


# ============== Model ==============

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, sigma: torch.Tensor) -> torch.Tensor:
        # sigma: (B,) or (B,1)
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0)
        if sigma.ndim == 2:
            sigma = sigma.squeeze(-1)

        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=sigma.device) / half)
        args = sigma[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, emb_dim: int, kernel_size: int = 7):
        super().__init__()
        pad = kernel_size // 2
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.emb_proj = nn.Linear(emb_dim, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.emb_proj(emb).unsqueeze(-1)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return x + h


class FreqDenoiser(nn.Module):
    """
    Denoises frequency-domain OFDM symbols.

    Input: noisy symbols (2 ch) + distorted observation (2 ch) = 4 channels
    Output: predicted clean symbols (2 ch)
    Conditioned on noise level sigma.
    """

    def __init__(self, base_ch: int = 64, emb_dim: int = 128, num_blocks: int = 6):
        super().__init__()

        self.sigma_embed = nn.Sequential(
            SinusoidalEmbedding(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # 4 input channels: [noisy_I, noisy_Q, distorted_I, distorted_Q]
        self.input_conv = nn.Conv1d(4, base_ch, 7, padding=3)

        self.blocks = nn.ModuleList([ResBlock1D(base_ch, emb_dim) for _ in range(num_blocks)])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(8, base_ch), base_ch),
            nn.GELU(),
            nn.Conv1d(base_ch, 2, 7, padding=3),
        )
        # Zero-init output for stable start
        nn.init.zeros_(self.output_conv[-1].weight)
        nn.init.zeros_(self.output_conv[-1].bias)

    def forward(self, x_noisy: torch.Tensor, cond: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_noisy: (B, 2, N_sub) noisy symbols
            cond:    (B, 2, N_sub) distorted symbols (conditioning)
            sigma:   (B,) noise level
        Returns:
            (B, 2, N_sub) predicted clean symbols
        """
        emb = self.sigma_embed(sigma)
        h = torch.cat([x_noisy, cond], dim=1)  # (B, 4, N_sub)
        h = self.input_conv(h)
        for block in self.blocks:
            h = block(h, emb)
        return self.output_conv(h)


# ============== Diffusion ==============

class FreqDiffusion:
    """Simple EDM-style diffusion for frequency-domain symbols."""

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 1.0, sigma_data: float = 0.5):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

    def sample_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        log_sigma = torch.rand(batch_size, device=device)
        log_sigma = log_sigma * (np.log(self.sigma_max) - np.log(self.sigma_min)) + np.log(self.sigma_min)
        return torch.exp(log_sigma)

    def get_scalings(self, sigma: torch.Tensor):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        return c_skip, c_out, c_in

    def get_schedule(self, num_steps: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0, 1, num_steps + 1, device=device)
        sigmas = self.sigma_max ** (1 - t) * self.sigma_min ** t
        sigmas[-1] = 0
        return sigmas


# ============== Data ==============

def generate_freq_domain_pair(config: OFDMConfig, signal_length: int, clip_level_mult: float):
    """
    Generate one OFDM signal, clip it, return freq-domain symbol pairs.

    Returns:
        clean_eq: (num_symbols, num_subcarriers) complex - equalized clean symbols
        distorted_eq: (num_symbols, num_subcarriers) complex - equalized distorted symbols
        metadata: OFDM metadata dict
    """
    signal, metadata = generate_ofdm_signal(config, signal_length, seed=None)

    clip_level = clip_level_mult * np.std(signal)
    clipped = clip_signal_np(signal, clip_level)

    # Demodulate
    clean_symbols = demodulate_ofdm(signal, metadata)
    distorted_symbols = demodulate_ofdm(clipped, metadata)

    # Equalize using gain estimated from CLEAN symbols (training convention)
    ref = metadata["data_symbols"]
    gain_clean = estimate_channel_gain(ref.flatten(), clean_symbols.flatten())

    clean_eq = clean_symbols / gain_clean
    distorted_eq = distorted_symbols / gain_clean

    return clean_eq, distorted_eq, metadata


# ============== Training ==============

def train(
    model: nn.Module,
    diffusion: FreqDiffusion,
    config: OFDMConfig,
    signal_length: int,
    clip_range,
    num_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup = min(200, num_steps // 10)
    losses = []

    pbar = tqdm(range(num_steps), desc="Training freq-domain diffusion")
    for step in pbar:
        # LR schedule
        if step < warmup:
            cur_lr = lr * (step + 1) / warmup
        else:
            progress = (step - warmup) / max(1, num_steps - warmup)
            cur_lr = lr * 0.5 * (1 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # Generate batch: multiple signals → pool all OFDM symbols, then sample B pairs
        all_clean = []
        all_distorted = []
        while len(all_clean) < batch_size:
            clip_mult = np.random.uniform(*clip_range)
            clean_eq, distorted_eq, _ = generate_freq_domain_pair(config, signal_length, clip_mult)
            all_clean.append(complex_to_2ch_np(clean_eq))       # (num_sym, 2, N_sub)
            all_distorted.append(complex_to_2ch_np(distorted_eq))

        all_clean = np.concatenate(all_clean, axis=0)          # (total_sym, 2, N_sub)
        all_distorted = np.concatenate(all_distorted, axis=0)

        idx = np.random.choice(len(all_clean), batch_size, replace=False)
        x_0 = torch.from_numpy(all_clean[idx]).to(device)       # (B, 2, N_sub)
        cond = torch.from_numpy(all_distorted[idx]).to(device)  # (B, 2, N_sub)

        # Diffusion training step
        B = x_0.shape[0]
        sigma = diffusion.sample_sigma(B, device)

        noise = torch.randn_like(x_0)
        x_noisy = x_0 + sigma[:, None, None] * noise

        c_skip, c_out, c_in = diffusion.get_scalings(sigma)
        model_input = c_in[:, None, None] * x_noisy
        model_output = model(model_input, cond, sigma)
        x_pred = c_skip[:, None, None] * x_noisy + c_out[:, None, None] * model_output

        loss = F.mse_loss(x_pred, x_0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if step % 200 == 0:
            avg = float(np.mean(losses[-200:])) if len(losses) >= 1 else 0.0
            pbar.set_postfix(loss=f"{avg:.5f}", lr=f"{cur_lr:.1e}")

    return losses


# ============== Inference ==============

@torch.no_grad()
def denoise_symbols(
    model: nn.Module,
    diffusion: FreqDiffusion,
    distorted: torch.Tensor,
    num_steps: int = 50,
    device: torch.device = torch.device("cpu"),
    s_churn: float = 0.4,
):
    """
    Denoise distorted frequency-domain symbols using stochastic diffusion.

    Uses EDM-style stochastic sampler with noise injection ("churn") to avoid
    mode-averaging in multimodal distributions like 16QAM/64QAM.

    Args:
        distorted: (B, 2, N_sub) distorted symbols as conditioning
        s_churn: Amount of stochastic noise injection (0 = deterministic, higher = more stochastic)
    Returns:
        (B, 2, N_sub) denoised symbols
    """
    model.eval()
    B, C, N = distorted.shape
    sigmas = diffusion.get_schedule(num_steps, device)

    # Warm start: begin from distorted symbols + noise
    x = distorted + torch.randn_like(distorted) * sigmas[0]

    for i in range(num_steps):
        sigma_t = sigmas[i]
        sigma_next = sigmas[i + 1]
        if sigma_t == 0:
            break

        # EDM churn
        gamma = min(s_churn / num_steps, np.sqrt(2) - 1)
        sigma_hat = sigma_t * (1 + gamma)
        if gamma > 0:
            noise = torch.randn_like(x)
            x = x + torch.sqrt(sigma_hat ** 2 - sigma_t ** 2) * noise

        sigma_batch = sigma_hat.expand(B)
        c_skip, c_out, c_in = diffusion.get_scalings(sigma_batch)

        model_input = c_in[:, None, None] * x
        model_output = model(model_input, distorted, sigma_batch)
        x_0_hat = c_skip[:, None, None] * x + c_out[:, None, None] * model_output

        # Euler step from sigma_hat to sigma_next
        if sigma_next > 0:
            d = (x - x_0_hat) / (sigma_hat + 1e-12)
            x = x + (sigma_next - sigma_hat) * d
        else:
            x = x_0_hat

    return x


# ============== Main ==============

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal_length", type=int, default=4096)
    parser.add_argument("--train_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--num_blocks", type=int, default=6)
    parser.add_argument("--clip_level", type=float, default=1.0,
                        help="Test clip level (multiple of std)")
    parser.add_argument("--clip_train_min", type=float, default=0.7,
                        help="Min clip level for training")
    parser.add_argument("--clip_train_max", type=float, default=1.5,
                        help="Max clip level for training")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--num_test", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="demo_results")
    parser.add_argument("--s_churn", type=float, default=0.8,
                        help="EDM churn (more helps 16QAM avoid averaging; try 0.8~2.0)")
    parser.add_argument("--do_snap", action="store_true",
                        help="Hard-snap diffusion output to constellation (recommended for 16QAM/64QAM)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = OFDMConfig()
    modulation = config.modulations[0]
    constellation = generate_qam_constellation(modulation)
    print(f"  Modulation: {modulation}")
    print(f"  Snap to constellation: {args.do_snap}")

    # ===== 1. Train =====
    print("=" * 60)
    print("Training frequency-domain diffusion model")
    print("=" * 60)

    # Figure out N_sub from a sample signal
    _, sample_meta = generate_ofdm_signal(config, args.signal_length, seed=0)
    n_sub = sample_meta["num_data_subcarriers"]
    print(f"  Subcarriers per OFDM symbol: {n_sub}")

    model = FreqDenoiser(base_ch=args.base_ch, emb_dim=128, num_blocks=args.num_blocks).to(device)
    diffusion = FreqDiffusion()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {num_params/1e6:.2f}M parameters")
    print(f"  Training clip range: [{args.clip_train_min}, {args.clip_train_max}] x std")
    print()

    losses = train(
        model, diffusion, config, args.signal_length,
        clip_range=(args.clip_train_min, args.clip_train_max),
        num_steps=args.train_steps, batch_size=args.batch_size,
        lr=args.lr, device=device,
    )

    # Plot training loss
    plt.figure(figsize=(10, 3))
    plt.plot(losses, alpha=0.3)
    if len(losses) > 50:
        smooth = np.convolve(losses, np.ones(50) / 50, mode="valid")
        plt.plot(np.arange(len(smooth)) + 25, smooth, "r-", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Frequency-Domain Diffusion Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "freq_diffusion_loss.png", dpi=150)
    plt.show()

    # ===== 2. Test =====
    print("\n" + "=" * 60)
    print(f"Testing declipping (clip_level={args.clip_level} x std)")
    print("=" * 60)

    all_evm_clip = []
    all_evm_diff = []

    for i in range(args.num_test):
        # Generate and clip
        signal, metadata = generate_ofdm_signal(config, args.signal_length, seed=1000 + i)
        clip_level = args.clip_level * np.std(signal)
        clipped = clip_signal_np(signal, clip_level)

        # Demodulate
        ref_symbols = metadata["data_symbols"]         # (num_sym, N_sub) complex
        ref_flat = ref_symbols.flatten()

        clean_demod = demodulate_ofdm(signal, metadata)    # (num_sym, N_sub)
        clip_demod = demodulate_ofdm(clipped, metadata)    # (num_sym, N_sub)

        # ===== FIX #1: use SAME equalization convention as training =====
        # gain from CLEAN, apply to BOTH clean and clipped
        gain_clean = estimate_channel_gain(ref_flat, clean_demod.flatten())
        orig_eq_flat = clean_demod.flatten() / gain_clean
        clip_eq = clip_demod / gain_clean

        evm_clip = compute_evm(ref_flat, clip_eq.flatten())

        # -- Diffusion declipping in frequency domain --
        # B = num_sym; each OFDM symbol is a "sample"
        clip_2ch = torch.from_numpy(complex_to_2ch_np(clip_eq)).to(device)  # (num_sym, 2, N_sub)

        denoised_2ch = denoise_symbols(
            model, diffusion, clip_2ch,
            num_steps=args.sampling_steps,
            device=device,
            s_churn=args.s_churn,
        )

        diff_symbols = ch2_to_complex_symbols(denoised_2ch)  # (num_sym, N_sub)
        diff_flat = diff_symbols.flatten()

        # Remove residual global gain (optional but helps)
        gain_diff = estimate_channel_gain(ref_flat, diff_flat)
        diff_eq_flat = diff_flat / gain_diff
        diff_eq_2d = diff_eq_flat.reshape(ref_symbols.shape)

        # ===== FIX #2: snap to constellation (highly recommended for 16QAM/64QAM) =====
        if args.do_snap:
            diff_eq_2d = snap_symbols_to_constellation(diff_eq_2d, constellation)
            diff_eq_flat_used = diff_eq_2d.flatten()
        else:
            diff_eq_flat_used = diff_eq_flat

        evm_diff = compute_evm(ref_flat, diff_eq_flat_used)

        all_evm_clip.append(evm_clip)
        all_evm_diff.append(evm_diff)

        print(
            f"  Signal {i+1}: clipped={evm_clip:.2f}%  →  diffusion={evm_diff:.2f}%  "
            f"(improvement: {evm_clip - evm_diff:.2f}%)"
        )

        # Remodulate diffusion output for time-domain plot
        recon_signal = remodulate_ofdm(diff_eq_2d, metadata)
        if len(recon_signal) < args.signal_length:
            recon_signal = np.pad(recon_signal, (0, args.signal_length - len(recon_signal)))
        else:
            recon_signal = recon_signal[:args.signal_length]
        recon_signal = recon_signal / np.std(recon_signal) * np.std(signal)

        # -- Plot: 2 rows x 3 cols per test signal --
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Signal {i+1} — Clip level: {args.clip_level}x std", fontsize=14)

        # Row 1: Constellations (original, clipped, declipped)
        axes[0, 0].scatter(orig_eq_flat.real, orig_eq_flat.imag, alpha=0.4, s=8, c="green")
        axes[0, 0].scatter(constellation.real, constellation.imag,
                           s=100, c="black", marker="x", linewidths=2, zorder=5)
        axes[0, 0].set_title(f"Original (EVM: {compute_evm(ref_flat, orig_eq_flat):.1f}%)")
        axes[0, 0].axis("equal"); axes[0, 0].set_xlim(-1.5, 1.5); axes[0, 0].set_ylim(-1.5, 1.5)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].scatter(clip_eq.real.flatten(), clip_eq.imag.flatten(), alpha=0.4, s=8, c="red")
        axes[0, 1].scatter(constellation.real, constellation.imag,
                           s=100, c="black", marker="x", linewidths=2, zorder=5)
        axes[0, 1].set_title(f"Clipped (EVM: {evm_clip:.1f}%)")
        axes[0, 1].axis("equal"); axes[0, 1].set_xlim(-1.5, 1.5); axes[0, 1].set_ylim(-1.5, 1.5)
        axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].scatter(diff_eq_flat_used.real, diff_eq_flat_used.imag, alpha=0.4, s=8, c="blue")
        axes[0, 2].scatter(constellation.real, constellation.imag,
                           s=100, c="black", marker="x", linewidths=2, zorder=5)
        axes[0, 2].set_title(f"Declipped (EVM: {evm_diff:.1f}%)")
        axes[0, 2].axis("equal"); axes[0, 2].set_xlim(-1.5, 1.5); axes[0, 2].set_ylim(-1.5, 1.5)
        axes[0, 2].grid(True, alpha=0.3)

        # Row 2: Time domain (original, clipped, declipped)
        t = np.arange(300)
        axes[1, 0].plot(t, signal.real[:300], "g-", alpha=0.7)
        axes[1, 0].plot(t, signal.imag[:300], "g--", alpha=0.5)
        axes[1, 0].set_title("Original (time domain)")
        axes[1, 0].set_xlabel("Sample")
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(t, clipped.real[:300], "r-", alpha=0.7)
        axes[1, 1].plot(t, clipped.imag[:300], "r--", alpha=0.5)
        axes[1, 1].axhline(y=clip_level, color="k", linestyle=":", alpha=0.3)
        axes[1, 1].axhline(y=-clip_level, color="k", linestyle=":", alpha=0.3)
        axes[1, 1].set_title("Clipped (time domain)")
        axes[1, 1].set_xlabel("Sample")
        axes[1, 1].grid(True, alpha=0.3)

        axes[1, 2].plot(t, recon_signal.real[:300], "b-", alpha=0.7)
        axes[1, 2].plot(t, recon_signal.imag[:300], "b--", alpha=0.5)
        axes[1, 2].set_title("Declipped (time domain)")
        axes[1, 2].set_xlabel("Sample")
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f"freq_diffusion_signal_{i+1}.png", dpi=150)
        plt.show()

    # ===== Summary =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Avg EVM clipped:    {np.mean(all_evm_clip):.2f}%")
    print(f"  Avg EVM diffusion:  {np.mean(all_evm_diff):.2f}%")
    print(f"  Avg improvement:    {np.mean(all_evm_clip) - np.mean(all_evm_diff):.2f}%")


if __name__ == "__main__":
    main()
