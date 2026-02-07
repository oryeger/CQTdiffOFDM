"""
Conditional Diffusion Learner for OFDM Declipping with EVM-focused Loss.

Training objective:
    L = L_eps + λ_f * L_freq + λ_m * L_mask

Where:
    - L_eps: Standard MSE on noise (diffusion objective)
    - L_freq: EVM-driven frequency loss (FFT domain) - weighted MSE on subcarriers
    - L_mask: Mask-weighted time loss emphasizing clipped regions

Features:
    - Conditional dropout for classifier-free guidance
    - Per-example RMS normalization (signal power = 1)
    - GroupNorm throughout (handles mixed FFT sizes)
"""

import os
import re
import time
import numpy as np
from glob import glob
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

from src.sde import VE_Sde_Elucidating
from src.ofdm.ofdm_generator import OFDMParams, demodulate_ofdm_torch
import src.utils.logging as utils_logging


class OFDMDeclipLearner:
    """
    Conditional diffusion learner for OFDM declipping with EVM-focused loss.
    
    Loss = L_eps + λ_f * ||W ⊙ (X̂ − X)||² + λ_m * ||m ⊙ (x̂₀ − x)||²
    
    Where:
        - L_eps: MSE on predicted noise ε̂
        - L_freq: Weighted frequency-domain loss after FFT
        - L_mask: Time-domain loss weighted by clipping mask
    """
    
    def __init__(
        self,
        model_dir: str,
        model: nn.Module,
        train_set,
        args,
        log: bool = True,
    ):
        """
        Args:
            model_dir: Directory for checkpoints and logs
            model: Conditional U-Net model
            train_set: DataLoader yielding (clean_signal, clipped_signal, mask, symbols, ofdm_params)
            args: Hydra configuration
            log: Whether to log to wandb
        """
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model
        self.device = next(self.model.parameters()).device
        
        self.step = 0
        
        if args.restore:
            self.restore_from_checkpoint()
        
        # EMA weights
        self.ema_weights = [p.clone().detach() for p in self.model.parameters()]
        
        # Diffusion parameters
        if args.sde_type == 'VE_elucidating':
            self.diff_params = VE_Sde_Elucidating(
                args.diffusion_parameters,
                args.diffusion_parameters.sigma_data
            )
        else:
            raise NotImplementedError(f"SDE type {args.sde_type} not implemented")
        
        self.args = args
        self.ema_rate = args.ema_rate
        self.train_set = train_set
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=getattr(args, 'weight_decay', 0.0)
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=args.scheduler_step_size,
            gamma=args.scheduler_gamma
        )
        
        # Loss weights
        self.lambda_freq = getattr(args, 'lambda_freq', 0.1)  # Frequency loss weight
        self.lambda_mask = getattr(args, 'lambda_mask', 0.1)  # Mask loss weight
        
        # Subcarrier weights for frequency loss (optional)
        self.weight_data_pilots = getattr(args, 'weight_data_pilots', 1.0)
        self.weight_nulled = getattr(args, 'weight_nulled', 0.0)
        
        print(f"Loss weights: λ_freq={self.lambda_freq}, λ_mask={self.lambda_mask}")
        
        # Logging
        self.log = log
        self.n_bins = args.n_bins
        self.accumulated_losses = None
        self.accumulated_losses_sigma = None
        self.cum_grad_norms = 0
        
        if self.log:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
            
            config_dict = {
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "sde_type": args.sde_type,
                "lambda_freq": self.lambda_freq,
                "lambda_mask": self.lambda_mask,
                "sigma_max": args.diffusion_parameters.sigma_max,
                "sigma_min": args.diffusion_parameters.sigma_min,
                "total_params": total_params,
            }
            wandb.init(project=args.wandb.project, entity=args.wandb.entity, config=config_dict)
            wandb.run.name = args.wandb.run_name + "_declip_" + wandb.run.id
    
    def normalize_signal(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize signal to unit RMS per example.
        
        Args:
            x: Signal, shape (B, 2, T) or (B, T)
        
        Returns:
            Normalized signal, scale factors
        """
        if x.ndim == 3:
            # Complex IQ: compute power across both channels and time
            rms = torch.sqrt(torch.mean(x ** 2, dim=(1, 2), keepdim=True) + 1e-8)
        else:
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
        
        return x / rms, rms
    
    def compute_noise_loss(
        self,
        eps_pred: torch.Tensor,
        eps_true: torch.Tensor,
    ) -> torch.Tensor:
        """Standard MSE loss on predicted noise."""
        return F.mse_loss(eps_pred, eps_true)
    
    def compute_frequency_loss(
        self,
        x_hat: torch.Tensor,
        x_clean: torch.Tensor,
        ofdm_params: OFDMParams,
        num_ofdm_symbols: int,
    ) -> torch.Tensor:
        """
        Compute EVM-driven frequency loss.
        
        Loss = ||W ⊙ (X̂ - X)||²
        
        Where:
            - X̂, X are FFT of predicted/clean signals (after CP removal)
            - W weights data/pilot subcarriers higher, optionally ignores nulled
        """
        B = x_hat.shape[0]
        
        # Convert to complex
        x_hat_complex = torch.complex(x_hat[:, 0, :], x_hat[:, 1, :])
        x_clean_complex = torch.complex(x_clean[:, 0, :], x_clean[:, 1, :])
        
        # Demodulate to get frequency-domain symbols
        X_hat = demodulate_ofdm_torch(x_hat_complex, ofdm_params, num_ofdm_symbols)
        X_clean = demodulate_ofdm_torch(x_clean_complex, ofdm_params, num_ofdm_symbols)
        
        # Compute weighted MSE
        error = X_hat - X_clean
        
        # Create subcarrier weights
        # Data subcarriers get weight_data_pilots, guard bands get weight_nulled
        num_subcarriers = X_hat.shape[-1]
        weights = torch.ones(num_subcarriers, device=x_hat.device) * self.weight_data_pilots
        
        # Optionally reduce weight on edge subcarriers (guard bands)
        num_guard = ofdm_params.num_guard_subcarriers
        if num_guard > 0 and self.weight_nulled < self.weight_data_pilots:
            # Note: In demodulate, we already extract only data subcarriers
            # So weights here apply to all extracted subcarriers
            pass
        
        # Weighted MSE
        weighted_error = weights * torch.abs(error) ** 2
        freq_loss = torch.mean(weighted_error)
        
        # Normalize by reference power
        ref_power = torch.mean(torch.abs(X_clean) ** 2) + 1e-8
        freq_loss = freq_loss / ref_power
        
        return freq_loss
    
    def compute_mask_loss(
        self,
        x_hat: torch.Tensor,
        x_clean: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute mask-weighted time-domain loss.
        
        Loss = ||m ⊙ (x̂₀ - x)||²
        
        Emphasizes reconstruction accuracy in clipped regions.
        """
        error = x_hat - x_clean
        
        # Expand mask to match signal shape if needed
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)  # (B, T) -> (B, 1, T)
        
        if mask.shape[1] == 1 and x_hat.shape[1] == 2:
            mask = mask.expand(-1, 2, -1)  # Expand to both I/Q channels
        
        # Weighted MSE (emphasize clipped samples)
        weighted_error = mask * (error ** 2)
        
        # Normalize by number of clipped samples
        num_clipped = torch.sum(mask) + 1e-8
        mask_loss = torch.sum(weighted_error) / num_clipped
        
        return mask_loss
    
    def get_x0_from_eps(
        self,
        x_t: torch.Tensor,
        eps: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute x₀ estimate from noise prediction.
        
        x_t = x_0 + sigma * eps
        => x_0 = x_t - sigma * eps
        """
        if sigma.ndim == 1:
            sigma = sigma.unsqueeze(-1).unsqueeze(-1)  # (B,) -> (B, 1, 1)
        elif sigma.ndim == 2:
            sigma = sigma.unsqueeze(-1)  # (B, 1) -> (B, 1, 1)
        
        return x_t - sigma * eps
    
    def train_step(self) -> Dict[str, torch.Tensor]:
        """
        Single training step with composite loss.
        
        Returns:
            Dictionary of loss components
        """
        self.model.train()
        
        for param in self.model.parameters():
            param.grad = None
        
        # Get batch: (clean, clipped, mask, symbols, ofdm_params, clip_level)
        batch = self.get_data_batch()
        x_clean, y_clipped, mask, symbols, ofdm_params, clip_level = batch
        
        B, C, T = x_clean.shape
        device = x_clean.device
        
        # Normalize to unit RMS
        x_clean_norm, scale = self.normalize_signal(x_clean)
        y_clipped_norm = y_clipped / scale
        
        # Sample noise levels
        sigma = self.diff_params.sample_ptrain_alt(B)
        sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        
        # Add noise to clean signal
        eps = torch.randn_like(x_clean_norm)
        x_t = x_clean_norm + sigma.unsqueeze(-1).unsqueeze(-1) * eps
        
        # Compute preconditioning (Karras)
        cnoise = self.diff_params.cnoise(sigma.unsqueeze(-1))
        
        # Forward pass
        eps_pred = self.model(
            x_t=x_t,
            t=cnoise,
            y=y_clipped_norm,
            m=mask,
            A=clip_level,
        )
        
        # ========== LOSS COMPUTATION ==========
        
        # 1. Noise prediction loss
        loss_eps = self.compute_noise_loss(eps_pred, eps)
        
        # 2. Frequency loss (EVM-focused)
        # Only compute for low-noise samples where x0 estimate is meaningful
        sigma_threshold = 0.5
        low_noise_mask = sigma < sigma_threshold
        
        if low_noise_mask.any() and self.lambda_freq > 0 and ofdm_params is not None:
            x_hat = self.get_x0_from_eps(x_t, eps_pred, sigma)
            
            # Denormalize for frequency loss computation
            x_hat_denorm = x_hat * scale
            x_clean_denorm = x_clean
            
            loss_freq = self.compute_frequency_loss(
                x_hat_denorm[low_noise_mask],
                x_clean_denorm[low_noise_mask],
                ofdm_params,
                symbols.shape[0] if symbols is not None else None,
            )
        else:
            loss_freq = torch.tensor(0.0, device=device)
        
        # 3. Mask-weighted time loss
        if self.lambda_mask > 0:
            x_hat = self.get_x0_from_eps(x_t, eps_pred, sigma)
            loss_mask = self.compute_mask_loss(x_hat, x_clean_norm, mask)
        else:
            loss_mask = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = loss_eps + self.lambda_freq * loss_freq + self.lambda_mask * loss_mask
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.cum_grad_norms += self.grad_norm
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        
        # Update EMA
        self.update_ema_weights()
        
        return {
            'total': total_loss,
            'eps': loss_eps,
            'freq': loss_freq,
            'mask': loss_mask,
            'sigma': sigma,
        }
    
    def get_data_batch(self):
        """
        Get batch from dataset.
        
        Expected format: (clean, clipped, mask, symbols, ofdm_params, clip_level)
        """
        batch = next(self.train_set)
        
        # Unpack batch
        if len(batch) == 6:
            x_clean, y_clipped, mask, symbols, ofdm_params, clip_level = batch
        elif len(batch) == 5:
            x_clean, y_clipped, mask, symbols, ofdm_params = batch
            clip_level = None
        elif len(batch) == 4:
            x_clean, y_clipped, mask, ofdm_params = batch
            symbols = None
            clip_level = None
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")
        
        # Move to device
        x_clean = x_clean.to(self.device)
        y_clipped = y_clipped.to(self.device)
        mask = mask.to(self.device)
        
        if clip_level is not None:
            clip_level = clip_level.to(self.device)
        
        return x_clean, y_clipped, mask, symbols, ofdm_params, clip_level
    
    def train(self):
        """Main training loop."""
        max_steps = getattr(self.args, 'max_steps', None)
        
        while True:
            start = time.time()
            
            losses = self.train_step()
            
            # Update accumulated losses for logging
            sigma_np = losses['sigma'].detach().cpu().numpy().flatten()
            loss_np = losses['eps'].detach().cpu().numpy()
            self.update_accumulated_loss(np.array([loss_np]), sigma_np)
            
            # Checkpointing
            if (self.step + 1) % self.args.save_interval == 0:
                if self.args.save_model:
                    self.save_to_checkpoint()
            
            # Logging
            if (self.step + 1) % self.args.log_interval == 0:
                if self.log:
                    self._write_summary(losses)
            
            self.step += 1
            elapsed = time.time() - start
            
            print(
                f"Step {self.step}: "
                f"loss={losses['total'].item():.5f} "
                f"(ε={losses['eps'].item():.5f}, "
                f"freq={losses['freq'].item():.5f}, "
                f"mask={losses['mask'].item():.5f}) "
                f"time={elapsed:.2f}s"
            )
            
            if max_steps is not None and self.step >= max_steps:
                print(f"\nReached max_steps ({max_steps}). Stopping.")
                if self.args.save_model:
                    self.save_to_checkpoint()
                break
    
    def update_ema_weights(self):
        """Update exponential moving average of model weights."""
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param.lerp_(param.detach(), 1 - self.ema_rate)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        if hasattr(self.model, "module"):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        return {
            "step": self.step,
            "model": {k: v.cpu() for k, v in model_state.items()},
            "ema_weights": [w.cpu() for w in self.ema_weights],
            "optimizer": self.optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        if hasattr(self.model, "module"):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        
        self.step = state_dict["step"]
        self.ema_weights = [w.to(self.device) for w in state_dict["ema_weights"]]
        
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
    
    def save_to_checkpoint(self, filename: str = "weights"):
        """Save checkpoint."""
        save_name = f"{self.model_dir}/{filename}-{self.step}.pt"
        torch.save(self.state_dict(), save_name)
        print(f"Saved checkpoint: {save_name}")
    
    def restore_from_checkpoint(self, checkpoint_id: Optional[int] = None) -> bool:
        """Restore from checkpoint."""
        try:
            if checkpoint_id is None:
                list_weights = glob(f'{self.model_dir}/weights-*')
                id_regex = re.compile(r'weights-(\d+)')
                list_ids = [
                    int(id_regex.search(w).groups()[0])
                    for w in list_weights
                ]
                checkpoint_id = max(list_ids)
            
            checkpoint = torch.load(
                f"{self.model_dir}/weights-{checkpoint_id}.pt",
                map_location=self.device
            )
            self.load_state_dict(checkpoint)
            print(f"Restored from checkpoint {checkpoint_id}")
            return True
        except (FileNotFoundError, ValueError) as e:
            print(f"Could not restore checkpoint: {e}")
            return False
    
    def update_accumulated_loss(self, loss: np.ndarray, sigma: np.ndarray):
        """Accumulate losses for logging."""
        if self.accumulated_losses is None:
            self.accumulated_losses = loss
            self.accumulated_losses_sigma = sigma
        else:
            self.accumulated_losses = np.concatenate([self.accumulated_losses, loss])
            self.accumulated_losses_sigma = np.concatenate([self.accumulated_losses_sigma, sigma])
    
    def _write_summary(self, losses: Dict[str, torch.Tensor]):
        """Write training summary to wandb."""
        if not self.log:
            return
        
        # Log scalar losses
        wandb.log({
            "loss/total": losses['total'].item(),
            "loss/eps": losses['eps'].item(),
            "loss/freq": losses['freq'].item(),
            "loss/mask": losses['mask'].item(),
            "grad_norm": self.grad_norm,
            "lr": self.scheduler.get_last_lr()[0],
        }, step=self.step)
        
        # Reset accumulated metrics
        self.cum_grad_norms = 0
        self.accumulated_losses = None
        self.accumulated_losses_sigma = None


class ClippingDataset(torch.utils.data.IterableDataset):
    """
    Dataset wrapper that applies MAGNITUDE-BASED clipping on-the-fly.
    
    Clipping is based on magnitude: if |x| = sqrt(I² + Q²) >= A,
    then scale the vector to have magnitude A while preserving phase.
    
    Yields: (clean_signal, clipped_signal, mask, symbols, ofdm_params, clip_level)
    """
    
    def __init__(
        self,
        base_dataset,
        clip_ratio_range: Tuple[float, float] = (0.3, 0.7),
        use_complex: bool = True,
    ):
        """
        Args:
            base_dataset: Dataset yielding (signal, symbols, ofdm_params) or just signal
            clip_ratio_range: Range of clip ratios to sample from
            use_complex: If True, signal is 2-channel (I, Q)
        """
        self.base_dataset = base_dataset
        self.clip_ratio_range = clip_ratio_range
        self.use_complex = use_complex
    
    def __iter__(self):
        for item in self.base_dataset:
            # Unpack item
            if isinstance(item, tuple) and len(item) == 3:
                signal, symbols, ofdm_params = item
            elif isinstance(item, tuple) and len(item) == 2:
                signal, symbols = item
                ofdm_params = None
            else:
                signal = item
                symbols = None
                ofdm_params = None
            
            # Convert to tensor if needed
            if isinstance(signal, np.ndarray):
                signal = torch.from_numpy(signal)
            
            # Ensure 2D: (2, T) for complex or (1, T) for real
            if signal.ndim == 1:
                if self.use_complex:
                    # Assume it's already real-valued OFDM, create I/Q representation
                    # For Hermitian-symmetric OFDM, I=signal, Q=Hilbert(signal) or zeros
                    signal = torch.stack([signal, torch.zeros_like(signal)], dim=0)
                else:
                    signal = signal.unsqueeze(0)
            
            # Random clip ratio
            clip_ratio = np.random.uniform(*self.clip_ratio_range)
            
            # Compute MAGNITUDE: |x| = sqrt(I² + Q²)
            I = signal[0]  # (T,)
            Q = signal[1]  # (T,)
            magnitude = torch.sqrt(I ** 2 + Q ** 2 + 1e-10)
            
            # Clip level based on peak magnitude
            peak = torch.max(magnitude)
            clip_level = peak * clip_ratio
            
            # Create mask: m = 1 where |x| >= A (BEFORE clipping)
            is_clipped = magnitude >= clip_level
            mask = is_clipped.float().unsqueeze(0)  # (1, T)
            
            # Apply MAGNITUDE clipping (preserve phase, limit magnitude)
            # x_clipped = x * min(1, A / |x|)
            scale = torch.ones_like(magnitude)
            scale[is_clipped] = clip_level / (magnitude[is_clipped] + 1e-10)
            
            clipped = torch.stack([
                I * scale,
                Q * scale
            ], dim=0)
            
            yield (
                signal.float(),
                clipped.float(),
                mask.float(),
                symbols,
                ofdm_params,
                torch.tensor([clip_level.item()]).float(),
            )


def create_clipping_dataloader(
    base_dataset,
    batch_size: int,
    clip_ratio_range: Tuple[float, float] = (0.3, 0.7),
    num_workers: int = 0,
):
    """Create DataLoader with on-the-fly clipping."""
    clipping_dataset = ClippingDataset(base_dataset, clip_ratio_range)
    
    def collate_fn(batch):
        signals, clipped, masks, symbols_list, params_list, clip_levels = zip(*batch)
        
        return (
            torch.stack(signals),
            torch.stack(clipped),
            torch.stack(masks),
            symbols_list[0] if symbols_list[0] is not None else None,
            params_list[0] if params_list[0] is not None else None,
            torch.stack(clip_levels),
        )
    
    return torch.utils.data.DataLoader(
        clipping_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
