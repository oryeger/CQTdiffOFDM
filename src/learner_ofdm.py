"""
OFDM-aware Learner with EVM loss.

This learner adds EVM (constellation) awareness during training by:
1. Computing standard denoising loss
2. Computing EVM loss on the demodulated symbols
3. Combining both losses with a configurable weight
"""

import numpy as np
import os
import re
import torch
import torchaudio
import torch.nn as nn

from tqdm import tqdm
from glob import glob
import time

from src.sampler import Sampler
import src.utils.logging as utils_logging
import wandb

from src.sde import VE_Sde_Elucidating
from src.ofdm.ofdm_generator import OFDMParams, demodulate_ofdm_torch


class LearnerOFDM:
    """
    OFDM-aware learner that includes EVM loss during training.
    
    The total loss is:
        loss = denoising_loss + lambda_evm * evm_loss
    
    Where:
        - denoising_loss: Standard MSE between predicted and target (Karras formulation)
        - evm_loss: MSE between demodulated symbols and reference symbols
        - lambda_evm: Weight for EVM loss (configurable)
    """
    
    def __init__(
        self, model_dir, model, train_set, args, log=True
    ):
        """
        Args:
            model_dir: Path where model weights and logs will be saved
            model: Neural network module
            train_set: DataLoader that yields (signal, symbols, ofdm_params) tuples
            args: Hydra configuration dictionary
            log: Whether to log to wandb
        """
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.model = model

        self.step = 0
        self.device = next(self.model.parameters()).device
        
        if args.restore:
            self.restore_from_checkpoint()

        self.ema_weights = [param.clone().detach() for param in self.model.parameters()]

        if args.sde_type == 'VE_elucidating':
            self.diff_parameters = VE_Sde_Elucidating(
                args.diffusion_parameters, 
                args.diffusion_parameters.sigma_data
            )
        else:
            raise NotImplementedError

        self.args = args
        self.sampler = Sampler(
            self.model, self.diff_parameters, self.args,
            xi=self.args.inference.xi,
            data_consistency=self.args.inference.data_consistency,
            rid=False
        )

        self.ema_rate = args.ema_rate
        self.train_set = train_set
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.args.scheduler_step_size, 
            gamma=self.args.scheduler_gamma
        )

        # Loss functions
        self.loss_fn = nn.MSELoss()
        self.v_loss = nn.MSELoss(reduction="none")

        # EVM loss weight - configurable via args
        self.lambda_evm = getattr(args, 'lambda_evm', 0.1)
        print(f"EVM loss weight (lambda_evm): {self.lambda_evm}")

        self.summary_writer = None
        self.n_bins = args.n_bins

        self.accumulated_losses = None
        self.accumulated_losses_sigma = None
        self.accumulated_evm_losses = None

        self.cum_grad_norms = 0
        self.log = log
        
        if self.log:
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("total_params: ", total_params/1e6, "M")
            
            config_dict = {
                "learning_rate": self.args.lr,
                "audio_len": self.args.audio_len,
                "sample_rate": self.args.sample_rate,
                "batch_size": self.args.batch_size,
                "dataset": self.args.dset.name,
                "sde_type": self.args.sde_type,
                "architecture": self.args.architecture,
                "lambda_evm": self.lambda_evm,
                "num_steps": args.inference.T,
                "ro": args.diffusion_parameters.ro,
                "sigma_max": args.diffusion_parameters.sigma_max,
                "sigma_min": args.diffusion_parameters.sigma_min,
                "total_params": total_params
            }
            wandb.init(project=args.wandb.project, entity=args.wandb.entity, config=config_dict)
            wandb.run.name = args.wandb.run_name + "_evm_" + wandb.run.id
            self.first_log = True

        S = self.args.resample_factor
        if S > 2.1 and S < 2.2:
            self.resample = torchaudio.transforms.Resample(160*2, 147).to(self.device)
        else:
            N = int(self.args.audio_len * S)
            self.resample = torchaudio.transforms.Resample(N, self.args.audio_len).to(self.device)

    def compute_evm_loss(self, predicted_signal, reference_symbols, ofdm_params):
        """
        Compute differentiable EVM loss.
        
        Args:
            predicted_signal: Denoised signal from model, shape (B, T) or (T,)
            reference_symbols: Original QAM symbols, shape (num_symbols, num_subcarriers)
            ofdm_params: OFDMParams object
            
        Returns:
            EVM loss (scalar tensor)
        """
        # Handle batch dimension
        if predicted_signal.ndim == 1:
            predicted_signal = predicted_signal.unsqueeze(0)
        
        batch_size = predicted_signal.shape[0]
        num_ofdm_symbols = reference_symbols.shape[0]
        
        total_evm_loss = 0.0
        
        for b in range(batch_size):
            # Demodulate predicted signal to get symbols
            recovered_symbols = demodulate_ofdm_torch(
                predicted_signal[b], 
                ofdm_params, 
                num_ofdm_symbols
            )
            
            # Convert reference to tensor if needed
            if isinstance(reference_symbols, np.ndarray):
                ref = torch.from_numpy(reference_symbols).to(predicted_signal.device)
            else:
                ref = reference_symbols.to(predicted_signal.device)
            
            # Ensure complex
            if not ref.is_complex():
                ref = torch.complex(ref.real, torch.zeros_like(ref))
            
            # Compute MSE between constellation points (EVM squared)
            error = recovered_symbols - ref
            evm_loss = torch.mean(torch.abs(error) ** 2)
            
            # Normalize by reference power for scale invariance
            ref_power = torch.mean(torch.abs(ref) ** 2) + 1e-10
            evm_loss = evm_loss / ref_power
            
            total_evm_loss = total_evm_loss + evm_loss
        
        return total_evm_loss / batch_size

    def state_dict(self):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            "step": self.step,
            "model": {
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in model_state.items()
            },
            'ema_weights': self.ema_weights,
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, "module") and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict["model"])
        else:
            self.model.load_state_dict(state_dict["model"])
        self.step = state_dict["step"]
        self.ema_weights = state_dict['ema_weights']

    def save_to_checkpoint(self, filename="weights"):
        save_basename = f"{filename}-{self.step}.pt"
        save_name = f"{self.model_dir}/{save_basename}"
        torch.save(self.state_dict(), save_name)

    def restore_from_checkpoint(self, checkpoint_id=None):
        try:
            if checkpoint_id is None:
                list_weights = glob(f'{self.model_dir}/weights-*')
                id_regex = re.compile('weights-(\d*)')
                list_ids = [int(id_regex.search(weight_path).groups()[0])
                            for weight_path in list_weights]
                checkpoint_id = max(list_ids)

            checkpoint = torch.load(
                f"{self.model_dir}/weights-{checkpoint_id}.pt", 
                map_location=self.device
            )
            self.load_state_dict(checkpoint)
            return True
        except (FileNotFoundError, ValueError):
            return False

    def sample(self):
        """Sample unconditional examples."""
        shape = (self.args.inference.unconditional.num_samples, self.args.audio_len)
        res = self.sampler.predict_unconditional(shape, self.device)
        self._write_summary_sample(res, "unconditional")

    def train(self):
        """Training loop with EVM loss."""
        max_steps = getattr(self.args, 'max_steps', None)
        
        while True:
            start = time.time()
            
            loss, denoising_loss, evm_loss, vectorial_loss, sigma = self.train_step()

            sigma_detach = sigma.clone().detach().cpu().numpy()
            sigma_detach = np.reshape(sigma_detach, -1)
            vectorial_loss_np = torch.mean(vectorial_loss, 1).cpu().numpy()
            vectorial_loss_np = np.reshape(vectorial_loss_np, -1)
            self.update_accumulated_loss(vectorial_loss_np, sigma_detach, True)

            if (self.step + 1) % self.args.save_interval == 0:
                if self.args.save_model:
                    self.save_to_checkpoint()
                if self.log:
                    self.sample()
                
            if (self.step + 1) % self.args.log_interval == 0:
                if self.log:
                    self._write_summary()

            self.step += 1
            end = time.time()

            print(f"Step: {self.step}, Loss: {loss.item():.6f} "
                  f"(denoise: {denoising_loss.item():.6f}, evm: {evm_loss.item():.6f}), "
                  f"Time: {end-start:.2f}s")
            
            # Check if we've reached max_steps
            if max_steps is not None and self.step >= max_steps:
                print(f"\nReached max_steps ({max_steps}). Stopping training.")
                if self.args.save_model:
                    self.save_to_checkpoint()
                    print(f"Final checkpoint saved.")
                break

    def get_data_batch(self):
        """Get one batch from dataset, including symbols and OFDM params."""
        batch = next(self.train_set)
        
        # Handle different return formats
        if isinstance(batch, tuple) and len(batch) == 3:
            # Format: (signal, symbols, ofdm_params)
            signal, symbols, ofdm_params = batch
        else:
            # Fallback: just signal (no EVM loss possible)
            signal = batch
            symbols = None
            ofdm_params = None
        
        signal = signal.to(self.device)
        
        if self.args.resample_factor != 1:
            signal = self.resample(signal)

        return signal, symbols, ofdm_params

    def train_step(self):
        """
        Training step with combined denoising + EVM loss.
        """
        for param in self.model.parameters():
            param.grad = None

        # Get data batch (now includes symbols and OFDM params)
        audio, symbols, ofdm_params = self.get_data_batch()

        N, T = audio.shape
        device = audio.device
        
        # Sample random noise levels
        sigma = self.diff_parameters.sample_ptrain_alt(N)
        sigma = torch.Tensor(sigma).to(audio.device)
        sigma = sigma.unsqueeze(-1)

        # Compute scaling parameters (Karras formulation)
        cskip = self.diff_parameters.cskip(sigma)
        cout = self.diff_parameters.cout(sigma)
        cin = self.diff_parameters.cin(sigma)
        cnoise = self.diff_parameters.cnoise(sigma)

        # Add noise
        noise = torch.randn_like(audio) * sigma
        noisy_audio = audio + noise

        # Forward pass
        estimate = self.model(cin * noisy_audio, cnoise)

        # Target (Karras Eq. 8)
        target = (1/cout) * (audio - cskip * noisy_audio)
        
        # ========== DENOISING LOSS ==========
        denoising_loss = self.loss_fn(estimate, target)
        
        # ========== EVM LOSS ==========
        if symbols is not None and ofdm_params is not None and self.lambda_evm > 0:
            # Reconstruct the denoised signal
            # D(x; sigma) = cskip * x + cout * F(cin * x; cnoise)
            denoised = cskip * noisy_audio + cout * estimate
            
            # For low noise levels, the denoised signal should be close to clean
            # Only compute EVM loss for samples with low sigma (where denoising is more accurate)
            sigma_threshold = 0.1  # Only compute EVM for low noise samples
            
            low_noise_mask = (sigma.squeeze() < sigma_threshold)
            
            if low_noise_mask.any():
                # Compute EVM loss only for low-noise samples
                low_noise_denoised = denoised[low_noise_mask]
                
                # Average EVM loss across the batch
                evm_loss = self.compute_evm_loss(
                    low_noise_denoised[0],  # Just use first sample for efficiency
                    symbols,
                    ofdm_params
                )
            else:
                evm_loss = torch.tensor(0.0, device=device)
        else:
            evm_loss = torch.tensor(0.0, device=device)

        # ========== COMBINED LOSS ==========
        total_loss = denoising_loss + self.lambda_evm * evm_loss

        # Backprop
        total_loss.backward()

        # Gradient clipping
        self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()

        # Update EMA weights
        self.update_ema_weights()

        # For logging
        vectorial_loss = self.v_loss(estimate, target).detach()
        self.cum_grad_norms += self.grad_norm

        return total_loss, denoising_loss, evm_loss, vectorial_loss, sigma

    def _write_summary_sample(self, res, string):
        """Log samples to wandb."""
        spec_sample = utils_logging.plot_spectrogram_from_raw_audio(res, self.args.stft)
        wandb.log({"spec_sample_" + str(string): spec_sample}, step=self.step)

        audio_path = utils_logging.write_audio_file(res, self.args.sample_rate, string)
        wandb.log({
            "audio_sample_" + str(string): wandb.Audio(audio_path, sample_rate=self.args.sample_rate)
        }, step=self.step)

    def _write_summary(self):
        """Log training summary to wandb."""
        sigma_max = self.diff_parameters.sigma_max
        sigma_min = self.diff_parameters.sigma_min
        quantized_sigma_values = self.diff_parameters.create_schedule(self.n_bins)
        ro = self.diff_parameters.ro
        sigma = self.accumulated_losses_sigma
        quantized_sigma = (sigma**(1/ro) - sigma_max**(1/ro)) * (self.n_bins-1) / (sigma_min**(1/ro) - sigma_max**(1/ro))
        quantized_sigma.astype(int)

        num_elems_in_bins = np.zeros(self.n_bins)
        sum_loss_in_bins = np.zeros(self.n_bins)

        for k in range(len(quantized_sigma)):
            i_bin = int(quantized_sigma[k])
            num_elems_in_bins[i_bin] += 1
            sum_loss_in_bins[i_bin] += self.accumulated_losses[k]
        
        figure = utils_logging.plot_loss_by_sigma_train(
            sum_loss_in_bins, num_elems_in_bins, quantized_sigma_values[:-1]
        )
        wandb.log({"loss_dependent_on_sigma": figure}, step=self.step)

        averaged_loss = np.mean(self.accumulated_losses)
        wandb.log({"averaged_loss": averaged_loss}, step=self.step)

        mean_grad_norms = self.cum_grad_norms / num_elems_in_bins.sum() * self.args.batch_size
        wandb.log({"mean_grad_norm": mean_grad_norms}, step=self.step, commit=True)

        self.cum_grad_norms = 0
        self.accumulated_losses = None
        self.accumulated_losses_sigma = None

    def update_accumulated_loss(self, vectorial_loss, sigma_array, isTrain):
        if (self.accumulated_losses is None) and (self.accumulated_losses_sigma is None):
            self.accumulated_losses = vectorial_loss
            self.accumulated_losses_sigma = sigma_array
        else:
            self.accumulated_losses = np.concatenate((self.accumulated_losses, vectorial_loss), axis=0)
            self.accumulated_losses_sigma = np.concatenate((self.accumulated_losses_sigma, sigma_array), axis=0)

    def update_ema_weights(self):
        for ema_param, param in zip(self.ema_weights, self.model.parameters()):
            if param.requires_grad:
                ema_param -= (1 - self.ema_rate) * (ema_param - param.detach())
            else:
                ema_param = param
