"""
Error Vector Magnitude (EVM) calculation utilities for OFDM signals.

EVM is defined as the RMS of the error vector normalized by the RMS of the reference signal:
EVM (%) = sqrt(mean(|error|^2)) / sqrt(mean(|reference|^2)) * 100

Where error = recovered_symbol - reference_symbol
"""

import numpy as np
import torch
from typing import Union, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_evm(
    reference_symbols: Union[np.ndarray, torch.Tensor],
    recovered_symbols: Union[np.ndarray, torch.Tensor],
    return_percentage: bool = True
) -> Union[float, torch.Tensor]:
    """
    Compute Error Vector Magnitude between reference and recovered symbols.

    Args:
        reference_symbols: Original QAM constellation points (complex)
        recovered_symbols: Recovered QAM constellation points (complex)
        return_percentage: If True, return EVM as percentage; otherwise return ratio

    Returns:
        EVM value (float for numpy, Tensor for torch)
    """
    is_torch = isinstance(reference_symbols, torch.Tensor)

    if is_torch:
        # Ensure both are complex tensors
        if not reference_symbols.is_complex():
            reference_symbols = torch.complex(reference_symbols, torch.zeros_like(reference_symbols))
        if not recovered_symbols.is_complex():
            recovered_symbols = torch.complex(recovered_symbols, torch.zeros_like(recovered_symbols))

        # Compute error vector
        error = recovered_symbols - reference_symbols

        # RMS of error
        error_rms = torch.sqrt(torch.mean(torch.abs(error) ** 2))

        # RMS of reference
        ref_rms = torch.sqrt(torch.mean(torch.abs(reference_symbols) ** 2))

        # EVM
        evm = error_rms / (ref_rms + 1e-10)

        if return_percentage:
            evm = evm * 100

        return evm

    else:
        # NumPy implementation
        reference_symbols = np.asarray(reference_symbols)
        recovered_symbols = np.asarray(recovered_symbols)

        # Compute error vector
        error = recovered_symbols - reference_symbols

        # RMS of error
        error_rms = np.sqrt(np.mean(np.abs(error) ** 2))

        # RMS of reference
        ref_rms = np.sqrt(np.mean(np.abs(reference_symbols) ** 2))

        # EVM
        evm = error_rms / (ref_rms + 1e-10)

        if return_percentage:
            evm = evm * 100

        return float(evm)


def compute_evm_db(
    reference_symbols: Union[np.ndarray, torch.Tensor],
    recovered_symbols: Union[np.ndarray, torch.Tensor]
) -> Union[float, torch.Tensor]:
    """
    Compute EVM in dB.

    Args:
        reference_symbols: Original QAM constellation points (complex)
        recovered_symbols: Recovered QAM constellation points (complex)

    Returns:
        EVM in dB
    """
    evm_ratio = compute_evm(reference_symbols, recovered_symbols, return_percentage=False)

    is_torch = isinstance(evm_ratio, torch.Tensor)

    if is_torch:
        return 20 * torch.log10(evm_ratio + 1e-10)
    else:
        return 20 * np.log10(evm_ratio + 1e-10)


def compute_evm_per_symbol(
    reference_symbols: Union[np.ndarray, torch.Tensor],
    recovered_symbols: Union[np.ndarray, torch.Tensor],
    return_percentage: bool = True
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute EVM for each OFDM symbol separately.

    Args:
        reference_symbols: Shape (num_ofdm_symbols, num_subcarriers) or (batch, num_ofdm_symbols, num_subcarriers)
        recovered_symbols: Same shape as reference_symbols
        return_percentage: If True, return EVM as percentage

    Returns:
        EVM per symbol, shape (num_ofdm_symbols,) or (batch, num_ofdm_symbols)
    """
    is_torch = isinstance(reference_symbols, torch.Tensor)

    if is_torch:
        # Ensure complex
        if not reference_symbols.is_complex():
            reference_symbols = torch.complex(reference_symbols, torch.zeros_like(reference_symbols))
        if not recovered_symbols.is_complex():
            recovered_symbols = torch.complex(recovered_symbols, torch.zeros_like(recovered_symbols))

        error = recovered_symbols - reference_symbols

        # RMS over subcarriers (last dimension)
        error_rms = torch.sqrt(torch.mean(torch.abs(error) ** 2, dim=-1))
        ref_rms = torch.sqrt(torch.mean(torch.abs(reference_symbols) ** 2, dim=-1))

        evm = error_rms / (ref_rms + 1e-10)

        if return_percentage:
            evm = evm * 100

        return evm

    else:
        reference_symbols = np.asarray(reference_symbols)
        recovered_symbols = np.asarray(recovered_symbols)

        error = recovered_symbols - reference_symbols

        # RMS over subcarriers (last dimension)
        error_rms = np.sqrt(np.mean(np.abs(error) ** 2, axis=-1))
        ref_rms = np.sqrt(np.mean(np.abs(reference_symbols) ** 2, axis=-1))

        evm = error_rms / (ref_rms + 1e-10)

        if return_percentage:
            evm = evm * 100

        return evm


def compute_evm_from_time_signal(
    reference_signal: Union[np.ndarray, torch.Tensor],
    recovered_signal: Union[np.ndarray, torch.Tensor],
    reference_symbols: Union[np.ndarray, torch.Tensor],
    ofdm_params,  # OFDMParams
    num_ofdm_symbols: Optional[int] = None,
    return_percentage: bool = True
) -> Union[float, torch.Tensor]:
    """
    Compute EVM by demodulating time-domain signals and comparing to reference symbols.

    Args:
        reference_signal: Original time-domain OFDM signal (for timing reference)
        recovered_signal: Recovered/declipped time-domain OFDM signal
        reference_symbols: Original QAM symbols used for modulation
        ofdm_params: OFDM parameters
        num_ofdm_symbols: Number of OFDM symbols
        return_percentage: If True, return EVM as percentage

    Returns:
        EVM value
    """
    is_torch = isinstance(recovered_signal, torch.Tensor)

    if is_torch:
        from src.ofdm.ofdm_generator import demodulate_ofdm_torch
        recovered_symbols = demodulate_ofdm_torch(recovered_signal, ofdm_params, num_ofdm_symbols)
    else:
        from src.ofdm.ofdm_generator import demodulate_ofdm
        recovered_symbols = demodulate_ofdm(recovered_signal, ofdm_params, num_ofdm_symbols)

    return compute_evm(reference_symbols, recovered_symbols, return_percentage)


def compute_evm_loss_torch(
    recovered_signal: torch.Tensor,
    reference_symbols: torch.Tensor,
    ofdm_params,  # OFDMParams
    num_ofdm_symbols: Optional[int] = None
) -> torch.Tensor:
    """
    Compute EVM as a differentiable loss for PyTorch optimization/guidance.

    This returns EVM as a ratio (not percentage) for better gradient scaling.

    Args:
        recovered_signal: Recovered time-domain signal, shape (batch, signal_length) or (signal_length,)
        reference_symbols: Original QAM symbols, shape (num_ofdm_symbols, num_subcarriers)
        ofdm_params: OFDM parameters
        num_ofdm_symbols: Number of OFDM symbols

    Returns:
        EVM loss (scalar tensor)
    """
    from src.ofdm.ofdm_generator import demodulate_ofdm_torch

    # Demodulate recovered signal
    recovered_symbols = demodulate_ofdm_torch(recovered_signal, ofdm_params, num_ofdm_symbols)

    # Handle batched vs unbatched
    if recovered_symbols.ndim == 2:  # unbatched: (num_symbols, num_subcarriers)
        ref = reference_symbols
    else:  # batched: (batch, num_symbols, num_subcarriers)
        # Expand reference to match batch
        ref = reference_symbols.unsqueeze(0).expand(recovered_symbols.shape[0], -1, -1)

    # Ensure complex tensors
    if not ref.is_complex():
        ref = torch.complex(ref.real, ref.imag if hasattr(ref, 'imag') else torch.zeros_like(ref))

    # Compute MSE in complex domain (this is what EVM^2 measures)
    error = recovered_symbols - ref
    mse = torch.mean(torch.abs(error) ** 2)

    # Normalize by reference power
    ref_power = torch.mean(torch.abs(ref) ** 2)

    # Return normalized MSE (EVM^2 essentially)
    # Using MSE instead of sqrt(MSE) for smoother gradients
    evm_squared = mse / (ref_power + 1e-10)

    return evm_squared
