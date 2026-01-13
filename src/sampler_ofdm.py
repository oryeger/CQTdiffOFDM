"""
OFDM-specific samplers with EVM guidance for declipping.

Extends the base sampler with:
- EVM-based reconstruction guidance
- Combined clipping consistency + EVM guidance
"""

from tqdm import tqdm
import torch
import numpy as np

from src.sampler import Sampler
from src.ofdm.ofdm_generator import OFDMParams, demodulate_ofdm_torch
from src.utils.evm import compute_evm, compute_evm_loss_torch


class SamplerOFDMDeclipping(Sampler):
    """
    Sampler for OFDM declipping with combined guidance:
    - Clipping consistency: ||clip(denoised) - y_clipped||
    - EVM guidance: minimize EVM between demodulated symbols and reference
    """

    def __init__(
        self,
        model,
        diff_params,
        args,
        xi=0,
        xi_evm=0,
        order=2,
        data_consistency=False,
        rid=False
    ):
        """
        Args:
            model: Diffusion model
            diff_params: Diffusion parameters
            args: Hydra arguments
            xi: Guidance weight for clipping consistency
            xi_evm: Guidance weight for EVM minimization
            order: ODE solver order (1 or 2)
            data_consistency: Whether to use data consistency step
            rid: Return intermediate denoised samples
        """
        super().__init__(model, diff_params, args, xi, order, data_consistency, rid)
        self.xi_evm = xi_evm

    def apply_clip(self, x):
        """Apply clipping degradation."""
        return torch.clip(x, min=-self.clip_value, max=self.clip_value)

    def get_score_combined_guidance(
        self,
        x,
        y_clipped,
        t_i,
        original_symbols,
        ofdm_params: OFDMParams
    ):
        """
        Compute score with combined clipping consistency and EVM guidance.

        Args:
            x: Current noisy sample, shape (B, T)
            y_clipped: Clipped observation, shape (B, T)
            t_i: Current timestep (noise level)
            original_symbols: Reference QAM symbols, shape (num_ofdm_symbols, num_subcarriers)
            ofdm_params: OFDM parameters

        Returns:
            Combined score
        """
        x.requires_grad_()

        # Denoise
        x_hat = self.diff_params.denoiser(x, self.model, t_i.unsqueeze(-1))

        # ====== Clipping Consistency Guidance ======
        if self.xi > 0:
            # Apply clipping to denoised estimate
            den_rec = self.apply_clip(x_hat)

            # Reconstruction error
            clip_error = y_clipped - den_rec

            if len(y_clipped.shape) == 2:
                dim = 1
            else:
                dim = (1, 2)

            clip_norm = torch.linalg.norm(clip_error, dim=dim, ord=2)

            # Compute gradients for clipping guidance
            clip_grads = torch.autograd.grad(
                outputs=clip_norm.sum(),
                inputs=x,
                retain_graph=True
            )[0]

            # Normalize
            clip_normguide = torch.linalg.norm(clip_grads) / (self.args.audio_len ** 0.5)
            s_clip = self.xi / (clip_normguide * t_i + 1e-6)

            # Apply threshold if specified
            if self.treshold_on_grads > 0:
                clip_grads = torch.clip(clip_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        else:
            clip_grads = 0
            s_clip = 0

        # ====== EVM Guidance ======
        if self.xi_evm > 0:
            # Demodulate to get constellation points
            num_ofdm_symbols = original_symbols.shape[0]
            recovered_symbols = demodulate_ofdm_torch(x_hat, ofdm_params, num_ofdm_symbols)

            # Convert original_symbols to tensor if needed
            if isinstance(original_symbols, np.ndarray):
                ref_symbols = torch.from_numpy(original_symbols).to(x.device)
            else:
                ref_symbols = original_symbols.to(x.device)

            # Handle batch dimension
            if x_hat.ndim == 1:
                # Unbatched
                pass
            else:
                # Batched: expand reference
                if recovered_symbols.ndim == 3:  # (batch, num_symbols, num_subcarriers)
                    ref_symbols = ref_symbols.unsqueeze(0).expand(recovered_symbols.shape[0], -1, -1)

            # EVM loss (squared EVM for smoother gradients)
            error = recovered_symbols - ref_symbols
            evm_loss = torch.mean(torch.abs(error) ** 2)

            # Compute gradients for EVM guidance
            evm_grads = torch.autograd.grad(
                outputs=evm_loss,
                inputs=x,
                retain_graph=False
            )[0]

            # Normalize
            evm_normguide = torch.linalg.norm(evm_grads) / (self.args.audio_len ** 0.5)
            s_evm = self.xi_evm / (evm_normguide * t_i + 1e-6)

            # Apply threshold if specified
            if self.treshold_on_grads > 0:
                evm_grads = torch.clip(evm_grads, min=-self.treshold_on_grads, max=self.treshold_on_grads)
        else:
            evm_grads = 0
            s_evm = 0

        # ====== Combine into Score ======
        # Base score from Tweedie's formula
        score = (x_hat.detach() - x) / t_i ** 2

        # Apply both guidances
        if self.xi > 0:
            score = score - s_clip * clip_grads
        if self.xi_evm > 0:
            score = score - s_evm * evm_grads

        return score

    def get_score(self, x, y, t_i, degradation):
        """
        Override base get_score to use combined guidance when OFDM parameters are set.
        """
        if not hasattr(self, '_ofdm_params') or self._original_symbols is None:
            # Fall back to base implementation
            return super().get_score(x, y, t_i, degradation)

        # Use combined guidance
        return self.get_score_combined_guidance(
            x, y, t_i, self._original_symbols, self._ofdm_params
        )

    def predict_ofdm_declipping(
        self,
        y_clipped,
        clip_value,
        original_symbols,
        ofdm_params: OFDMParams
    ):
        """
        Predict declipped OFDM signal with EVM guidance.

        Args:
            y_clipped: Clipped OFDM signal, shape (B, T) or (T,)
            clip_value: Clipping threshold
            original_symbols: Original QAM symbols for EVM reference
            ofdm_params: OFDM parameters

        Returns:
            Declipped signal
        """
        self.clip_value = clip_value
        self._original_symbols = original_symbols
        self._ofdm_params = ofdm_params

        # Define degradation function
        degradation = lambda x: self.apply_clip(x)
        self.degradation = degradation
        self.y = y_clipped

        # Run diffusion sampling
        if self.rid:
            res, denoised, t = self.predict(y_clipped.shape, y_clipped.device)
            return res, denoised, t
        else:
            res = self.predict(y_clipped.shape, y_clipped.device)
            return res

    def predict(self, shape, device):
        """
        Override predict to use combined guidance.
        """
        if self.rid:
            data_denoised = torch.zeros((self.nb_steps, shape[0], shape[1]))

        # Get the noise schedule
        t = self.diff_params.create_schedule(self.nb_steps).to(device)

        # Sample from gaussian distribution with sigma_max variance
        x = self.diff_params.sample_prior(shape, t[0]).to(device)

        # Parameter for Langevin stochasticity
        gamma = self.diff_params.get_gamma(t).to(device)

        for i in tqdm(range(0, self.nb_steps, 1)):
            if gamma[i] == 0:
                # Deterministic sampling
                t_hat = t[i]
                x_hat = x
            else:
                # Stochastic sampling
                t_hat = t[i] + gamma[i] * t[i]
                epsilon = torch.randn(shape).to(device) * self.diff_params.Snoise
                x_hat = x + ((t_hat ** 2 - t[i] ** 2) ** (1 / 2)) * epsilon

            # Get score with combined guidance
            if hasattr(self, '_ofdm_params') and self._original_symbols is not None:
                score = self.get_score_combined_guidance(
                    x_hat, self.y, t_hat, self._original_symbols, self._ofdm_params
                )
            else:
                score = self.get_score(x_hat, self.y, t_hat, self.degradation)

            d = -t_hat * score

            # Apply ODE step
            h = t[i + 1] - t_hat

            if t[i + 1] != 0 and self.order == 2:
                # Second order correction
                t_prime = t[i + 1]
                x_prime = x_hat + h * d

                if hasattr(self, '_ofdm_params') and self._original_symbols is not None:
                    score_prime = self.get_score_combined_guidance(
                        x_prime, self.y, t_prime, self._original_symbols, self._ofdm_params
                    )
                else:
                    score_prime = self.get_score(x_prime, self.y, t_prime, self.degradation)

                d_prime = -t_prime * score_prime
                x = x_hat + h * ((1 / 2) * d + (1 / 2) * d_prime)
            else:
                # First order Euler step
                x = x_hat + h * d

            if self.rid:
                data_denoised[i] = x

        if self.rid:
            return x.detach(), data_denoised.detach(), t.detach()
        else:
            return x.detach()


class SamplerOFDMDeclippingEVMOnly(SamplerOFDMDeclipping):
    """
    Sampler using only EVM guidance (no clipping consistency).
    """

    def __init__(self, model, diff_params, args, xi_evm=1.0, order=2, rid=False):
        super().__init__(
            model, diff_params, args,
            xi=0,  # No clipping guidance
            xi_evm=xi_evm,
            order=order,
            data_consistency=False,
            rid=rid
        )


class SamplerOFDMDeclippingClipOnly(SamplerOFDMDeclipping):
    """
    Sampler using only clipping consistency (no EVM guidance).
    Same as original audio declipping sampler.
    """

    def __init__(self, model, diff_params, args, xi=1.0, order=2, data_consistency=False, rid=False):
        super().__init__(
            model, diff_params, args,
            xi=xi,
            xi_evm=0,  # No EVM guidance
            order=order,
            data_consistency=data_consistency,
            rid=rid
        )
