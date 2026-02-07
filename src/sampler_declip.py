"""
DDIM Sampler with Data Consistency for OFDM Declipping.

Implements inpainting diffusion:
- For unmasked (unclipped) samples, overwrite x_t[~m] with forward-noised y[~m]
- This enforces data consistency: unclipped regions stay faithful to observation

Features:
- DDIM sampling (20-50 steps) for speed
- Optional classifier-free guidance
- Data consistency via inpainting at each step
"""

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple, Callable, Union

from src.sde import VE_Sde_Elucidating
from src.ofdm.ofdm_generator import OFDMParams, demodulate_ofdm_torch
from src.utils.evm import compute_evm


class DDIMSamplerDeclip:
    """
    DDIM sampler with data consistency for OFDM declipping.
    
    At each sampling step:
    1. Denoise x_t to get x_0 estimate
    2. For unclipped samples (m=0), replace x_0 estimate with observation y
    3. Re-noise to get x_{t-1}
    
    This is the "replacement" method for inpainting diffusion.
    """
    
    def __init__(
        self,
        model: nn.Module,
        diff_params: VE_Sde_Elucidating,
        num_steps: int = 50,
        eta: float = 0.0,  # DDIM parameter (0 = deterministic)
        cfg_scale: float = 1.0,  # Classifier-free guidance scale
        data_consistency: bool = True,  # Enable inpainting
        guidance_weight: float = 0.0,  # Additional reconstruction guidance
    ):
        """
        Args:
            model: Conditional diffusion model
            diff_params: Diffusion parameters (VE_Sde_Elucidating)
            num_steps: Number of sampling steps (20-50 recommended)
            eta: DDIM stochasticity parameter (0 = deterministic DDIM)
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            data_consistency: Whether to apply inpainting data consistency
            guidance_weight: Weight for additional reconstruction guidance
        """
        self.model = model
        self.diff_params = diff_params
        self.num_steps = num_steps
        self.eta = eta
        self.cfg_scale = cfg_scale
        self.data_consistency = data_consistency
        self.guidance_weight = guidance_weight
    
    def get_schedule(self, device: torch.device) -> torch.Tensor:
        """Get sigma schedule for sampling."""
        return self.diff_params.create_schedule(self.num_steps).to(device)
    
    @torch.no_grad()
    def denoise(
        self,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply denoiser to get x_0 estimate.
        
        D(x_t, σ) = x_t - σ * ε̂(x_t, σ, y, m)
        
        For EDM/Karras parameterization.
        """
        # Get scaling factors
        sigma_unsq = sigma.unsqueeze(-1).unsqueeze(-1) if sigma.ndim == 1 else sigma.unsqueeze(-1)
        
        c_skip = self.diff_params.cskip(sigma_unsq)
        c_out = self.diff_params.cout(sigma_unsq)
        c_in = self.diff_params.cin(sigma_unsq)
        c_noise = self.diff_params.cnoise(sigma_unsq)
        
        # Get noise prediction
        if self.cfg_scale != 1.0 and hasattr(self.model, 'forward_with_cfg'):
            eps = self.model.forward_with_cfg(
                c_in * x_t, c_noise.squeeze(-1), y, m, A, cfg_scale=self.cfg_scale
            )
        else:
            eps = self.model(c_in * x_t, c_noise.squeeze(-1), y, m, A)
        
        # Compute x_0 estimate
        x_0 = c_skip * x_t + c_out * eps
        
        return x_0, eps
    
    def apply_data_consistency(
        self,
        x_0_pred: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply data consistency via inpainting.
        
        For unclipped samples (m=0), replace x_0 prediction with observation y.
        
        x_0_consistent[~m] = y[~m]
        x_0_consistent[m] = x_0_pred[m]
        """
        # Expand mask if needed
        if m.ndim == 3 and m.shape[1] == 1:
            m_expanded = m.expand(-1, x_0_pred.shape[1], -1)
        elif m.ndim == 2:
            m_expanded = m.unsqueeze(1).expand(-1, x_0_pred.shape[1], -1)
        else:
            m_expanded = m
        
        # Replace unclipped samples with observation
        x_0_consistent = torch.where(m_expanded > 0.5, x_0_pred, y)
        
        return x_0_consistent
    
    def add_noise(
        self,
        x_0: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to get x_t: x_t = x_0 + σ * ε"""
        eps = torch.randn_like(x_0)
        
        if sigma.ndim == 0:
            sigma = sigma.view(1, 1, 1)
        elif sigma.ndim == 1:
            sigma = sigma.view(-1, 1, 1)
        
        return x_0 + sigma * eps, eps
    
    @torch.no_grad()
    def sample_step(
        self,
        x_t: torch.Tensor,
        t_curr: torch.Tensor,
        t_next: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single DDIM sampling step with data consistency.
        
        DDIM update (deterministic when eta=0):
        x_{t-1} = sqrt(α_{t-1}) * x_0_pred + sqrt(1-α_{t-1}-σ²_t) * ε_pred + σ_t * ε
        
        For VE-SDE with σ as time:
        x_{t-1} = x_0_pred + σ_{t-1} * direction
        
        With data consistency:
        x_0_pred[~m] = y[~m] before computing x_{t-1}
        """
        # Denoise to get x_0 estimate
        x_0_pred, eps_pred = self.denoise(x_t, t_curr, y, m, A)
        
        # Apply data consistency (inpainting)
        if self.data_consistency:
            x_0_pred = self.apply_data_consistency(x_0_pred, y, m)
        
        # If at final step, return x_0
        if t_next == 0:
            return x_0_pred
        
        # DDIM step
        # direction = (x_t - x_0_pred) / t_curr
        if t_curr.ndim == 0:
            t_curr_expanded = t_curr.view(1, 1, 1)
            t_next_expanded = t_next.view(1, 1, 1)
        else:
            t_curr_expanded = t_curr.view(-1, 1, 1)
            t_next_expanded = t_next.view(-1, 1, 1)
        
        # Compute direction (predicted noise scaled)
        direction = (x_t - x_0_pred) / (t_curr_expanded + 1e-8)
        
        # DDIM update
        if self.eta == 0:
            # Deterministic DDIM
            x_next = x_0_pred + t_next_expanded * direction
        else:
            # Stochastic DDIM
            # σ_ddim = η * sqrt((σ²_{t-1}/σ²_t) * (1 - σ²_t/σ²_{t-1}))
            sigma_ddim = self.eta * torch.sqrt(
                (t_next_expanded ** 2 / t_curr_expanded ** 2) *
                (t_curr_expanded ** 2 - t_next_expanded ** 2)
            )
            
            # Predicted x_{t-1} component
            x_pred = x_0_pred + torch.sqrt(t_next_expanded ** 2 - sigma_ddim ** 2) * direction
            
            # Add noise
            noise = torch.randn_like(x_t)
            x_next = x_pred + sigma_ddim * noise
        
        return x_next
    
    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
        return_trajectory: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, list]]:
        """
        Sample declipped signal using DDIM with data consistency.
        
        Args:
            y: Clipped observation, shape (B, 2, T)
            m: Clipping mask (1 where clipped), shape (B, 1, T) or (B, T)
            A: Optional clip level, shape (B,) or (B, 1)
            return_trajectory: If True, return intermediate samples
        
        Returns:
            Declipped signal, shape (B, 2, T)
            Optionally, list of intermediate samples
        """
        device = y.device
        B, C, T = y.shape
        
        # Get noise schedule
        sigmas = self.get_schedule(device)
        
        # Initialize from pure noise scaled by sigma_max
        x_t = torch.randn(B, C, T, device=device) * sigmas[0]
        
        # Apply initial data consistency
        if self.data_consistency:
            # For unclipped regions, initialize closer to observation
            x_noisy, _ = self.add_noise(y, sigmas[0])
            
            if m.ndim == 2:
                m_expanded = m.unsqueeze(1).expand(-1, C, -1)
            elif m.shape[1] == 1:
                m_expanded = m.expand(-1, C, -1)
            else:
                m_expanded = m
            
            x_t = torch.where(m_expanded > 0.5, x_t, x_noisy)
        
        trajectory = [x_t.clone()] if return_trajectory else None
        
        # Sampling loop
        for i in tqdm(range(self.num_steps), desc="DDIM Sampling"):
            t_curr = sigmas[i]
            t_next = sigmas[i + 1]
            
            x_t = self.sample_step(x_t, t_curr, t_next, y, m, A)
            
            if return_trajectory:
                trajectory.append(x_t.clone())
        
        if return_trajectory:
            return x_t, trajectory
        return x_t
    
    @torch.no_grad()
    def sample_with_guidance(
        self,
        y: torch.Tensor,
        m: torch.Tensor,
        clip_value: float,
        A: Optional[torch.Tensor] = None,
        guidance_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """
        Sample with additional reconstruction guidance.
        
        Adds gradient guidance to encourage:
        - Clipping consistency: ||clip(x_0) - y||²
        - Optional EVM minimization
        """
        device = y.device
        B, C, T = y.shape
        
        sigmas = self.get_schedule(device)
        x_t = torch.randn(B, C, T, device=device) * sigmas[0]
        
        for i in tqdm(range(self.num_steps), desc="Guided Sampling"):
            t_curr = sigmas[i]
            t_next = sigmas[i + 1]
            
            if self.guidance_weight > 0 and guidance_fn is not None:
                # Enable gradients for guidance
                x_t.requires_grad_(True)
                
                # Get x_0 prediction
                x_0_pred, _ = self.denoise(x_t, t_curr, y, m, A)
                
                # Compute guidance gradient
                guidance_loss = guidance_fn(x_0_pred, y, m, clip_value)
                grad = torch.autograd.grad(guidance_loss, x_t)[0]
                
                x_t = x_t.detach()
                
                # Apply guidance
                x_t = x_t - self.guidance_weight * grad * t_curr
            
            # Standard DDIM step
            x_t = self.sample_step(x_t, t_curr, t_next, y, m, A)
        
        return x_t


class DDPMSamplerDeclip:
    """
    DDPM sampler with data consistency (alternative to DDIM).
    
    Uses stochastic sampling with full noise schedule.
    Slower than DDIM but can produce higher quality in some cases.
    """
    
    def __init__(
        self,
        model: nn.Module,
        diff_params: VE_Sde_Elucidating,
        num_steps: int = 100,
        cfg_scale: float = 1.0,
        data_consistency: bool = True,
    ):
        self.model = model
        self.diff_params = diff_params
        self.num_steps = num_steps
        self.cfg_scale = cfg_scale
        self.data_consistency = data_consistency
    
    @torch.no_grad()
    def sample(
        self,
        y: torch.Tensor,
        m: torch.Tensor,
        A: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample using DDPM with data consistency."""
        device = y.device
        B, C, T = y.shape
        
        sigmas = self.diff_params.create_schedule(self.num_steps).to(device)
        gamma = self.diff_params.get_gamma(sigmas)
        
        x = torch.randn(B, C, T, device=device) * sigmas[0]
        
        for i in tqdm(range(self.num_steps), desc="DDPM Sampling"):
            # Stochasticity parameter
            if gamma[i] > 0:
                t_hat = sigmas[i] + gamma[i] * sigmas[i]
                noise = torch.randn_like(x) * self.diff_params.Snoise
                x_hat = x + torch.sqrt(t_hat ** 2 - sigmas[i] ** 2) * noise
            else:
                t_hat = sigmas[i]
                x_hat = x
            
            # Get scalings
            t_hat_tensor = torch.full((B,), t_hat.item(), device=device)
            
            c_skip = self.diff_params.cskip(t_hat_tensor.unsqueeze(-1))
            c_out = self.diff_params.cout(t_hat_tensor.unsqueeze(-1))
            c_in = self.diff_params.cin(t_hat_tensor.unsqueeze(-1))
            c_noise = self.diff_params.cnoise(t_hat_tensor.unsqueeze(-1))
            
            # Denoise
            if self.cfg_scale != 1.0 and hasattr(self.model, 'forward_with_cfg'):
                eps = self.model.forward_with_cfg(
                    c_in.unsqueeze(-1) * x_hat, c_noise, y, m, A, cfg_scale=self.cfg_scale
                )
            else:
                eps = self.model(c_in.unsqueeze(-1) * x_hat, c_noise, y, m, A)
            
            x_0 = c_skip.unsqueeze(-1) * x_hat + c_out.unsqueeze(-1) * eps
            
            # Data consistency
            if self.data_consistency:
                if m.ndim == 2:
                    m_exp = m.unsqueeze(1).expand(-1, C, -1)
                elif m.shape[1] == 1:
                    m_exp = m.expand(-1, C, -1)
                else:
                    m_exp = m
                x_0 = torch.where(m_exp > 0.5, x_0, y)
            
            # Score
            score = (x_0 - x_hat) / (t_hat ** 2)
            
            # Euler step
            d = -t_hat * score
            h = sigmas[i + 1] - t_hat
            
            if sigmas[i + 1] != 0:
                # 2nd order correction
                x_prime = x_hat + h * d
                
                # Re-evaluate score at x_prime
                t_prime = sigmas[i + 1]
                t_prime_tensor = torch.full((B,), t_prime.item(), device=device)
                
                c_skip_p = self.diff_params.cskip(t_prime_tensor.unsqueeze(-1))
                c_out_p = self.diff_params.cout(t_prime_tensor.unsqueeze(-1))
                c_in_p = self.diff_params.cin(t_prime_tensor.unsqueeze(-1))
                c_noise_p = self.diff_params.cnoise(t_prime_tensor.unsqueeze(-1))
                
                if self.cfg_scale != 1.0 and hasattr(self.model, 'forward_with_cfg'):
                    eps_p = self.model.forward_with_cfg(
                        c_in_p.unsqueeze(-1) * x_prime, c_noise_p, y, m, A, cfg_scale=self.cfg_scale
                    )
                else:
                    eps_p = self.model(c_in_p.unsqueeze(-1) * x_prime, c_noise_p, y, m, A)
                
                x_0_p = c_skip_p.unsqueeze(-1) * x_prime + c_out_p.unsqueeze(-1) * eps_p
                
                if self.data_consistency:
                    x_0_p = torch.where(m_exp > 0.5, x_0_p, y)
                
                score_p = (x_0_p - x_prime) / (t_prime ** 2 + 1e-8)
                d_prime = -t_prime * score_p
                
                x = x_hat + h * (0.5 * d + 0.5 * d_prime)
            else:
                x = x_0
        
        return x


def magnitude_clip(x: torch.Tensor, clip_value: float) -> torch.Tensor:
    """
    Apply MAGNITUDE-BASED clipping to complex IQ signal.
    
    If |x| = sqrt(I² + Q²) >= A, scale to magnitude A while preserving phase.
    
    Args:
        x: Signal tensor, shape (B, 2, T) with [I, Q] channels
        clip_value: Clipping threshold A
    
    Returns:
        Clipped signal, shape (B, 2, T)
    """
    I = x[:, 0, :]  # (B, T)
    Q = x[:, 1, :]  # (B, T)
    magnitude = torch.sqrt(I ** 2 + Q ** 2 + 1e-10)
    
    # Scale factor: min(1, A / |x|)
    scale = torch.clamp(clip_value / magnitude, max=1.0)
    
    I_clipped = I * scale
    Q_clipped = Q * scale
    
    return torch.stack([I_clipped, Q_clipped], dim=1)


def get_magnitude_mask(x: torch.Tensor, clip_value: float) -> torch.Tensor:
    """
    Get clipping mask based on magnitude: m = 1 where sqrt(I² + Q²) >= A.
    
    Args:
        x: Signal tensor, shape (B, 2, T) with [I, Q] channels
        clip_value: Clipping threshold A
    
    Returns:
        Mask tensor, shape (B, 1, T)
    """
    magnitude = torch.sqrt(x[:, 0, :] ** 2 + x[:, 1, :] ** 2)
    return (magnitude >= clip_value).float().unsqueeze(1)


def reconstruction_guidance_fn(
    x_0: torch.Tensor,
    y: torch.Tensor,
    m: torch.Tensor,
    clip_value: float,
) -> torch.Tensor:
    """
    Reconstruction guidance for declipping.
    
    Encourages: magnitude_clip(x_0) ≈ y_clipped
    """
    x_clipped = magnitude_clip(x_0, clip_value)
    return torch.mean((x_clipped - y) ** 2)


def evm_guidance_fn(
    x_0: torch.Tensor,
    reference_symbols: torch.Tensor,
    ofdm_params: OFDMParams,
) -> torch.Tensor:
    """
    EVM-based guidance for constellation accuracy.
    
    Minimizes EVM between demodulated symbols and reference.
    """
    # Convert to complex
    x_complex = torch.complex(x_0[:, 0, :], x_0[:, 1, :])
    
    # Demodulate
    num_symbols = reference_symbols.shape[0]
    recovered = demodulate_ofdm_torch(x_complex, ofdm_params, num_symbols)
    
    # EVM loss
    ref = reference_symbols.to(x_0.device)
    error = recovered - ref
    return torch.mean(torch.abs(error) ** 2)


class CombinedGuidanceFn:
    """Combined reconstruction + EVM guidance."""
    
    def __init__(
        self,
        clip_value: float,
        ofdm_params: Optional[OFDMParams] = None,
        reference_symbols: Optional[torch.Tensor] = None,
        weight_recon: float = 1.0,
        weight_evm: float = 0.0,
    ):
        self.clip_value = clip_value
        self.ofdm_params = ofdm_params
        self.reference_symbols = reference_symbols
        self.weight_recon = weight_recon
        self.weight_evm = weight_evm
    
    def __call__(
        self,
        x_0: torch.Tensor,
        y: torch.Tensor,
        m: torch.Tensor,
        clip_value: float,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=x_0.device)
        
        if self.weight_recon > 0:
            loss = loss + self.weight_recon * reconstruction_guidance_fn(
                x_0, y, m, clip_value
            )
        
        if self.weight_evm > 0 and self.ofdm_params is not None and self.reference_symbols is not None:
            loss = loss + self.weight_evm * evm_guidance_fn(
                x_0, self.reference_symbols, self.ofdm_params
            )
        
        return loss


# Convenience function
def create_ddim_sampler(
    model: nn.Module,
    diff_params: VE_Sde_Elucidating,
    num_steps: int = 50,
    cfg_scale: float = 1.5,
    data_consistency: bool = True,
) -> DDIMSamplerDeclip:
    """Create DDIM sampler with recommended settings for OFDM declipping."""
    return DDIMSamplerDeclip(
        model=model,
        diff_params=diff_params,
        num_steps=num_steps,
        eta=0.0,  # Deterministic DDIM
        cfg_scale=cfg_scale,
        data_consistency=data_consistency,
    )
