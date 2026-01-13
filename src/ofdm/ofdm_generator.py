"""
OFDM Signal Generator with Hermitian-symmetric frequency domain
for real-valued time domain signals.

Supports:
- Modulation: QPSK, 16QAM, 64QAM, 256QAM
- Configurable FFT size, guard band ratio, and cyclic prefix ratio
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Literal

ModulationType = Literal['QPSK', '16QAM', '64QAM', '256QAM']

# QAM constellation definitions (normalized to unit average power)
def _create_qam_constellation(M: int) -> np.ndarray:
    """Create M-QAM constellation with unit average power."""
    sqrt_M = int(np.sqrt(M))
    assert sqrt_M ** 2 == M, "M must be a perfect square"

    # Create grid
    points = np.arange(sqrt_M) - (sqrt_M - 1) / 2
    real, imag = np.meshgrid(points, points)
    constellation = (real + 1j * imag).flatten()

    # Normalize to unit average power
    avg_power = np.mean(np.abs(constellation) ** 2)
    constellation = constellation / np.sqrt(avg_power)

    return constellation

QAM_CONSTELLATIONS = {
    'QPSK': _create_qam_constellation(4),
    '16QAM': _create_qam_constellation(16),
    '64QAM': _create_qam_constellation(64),
    '256QAM': _create_qam_constellation(256),
}

@dataclass
class OFDMParams:
    """Parameters for OFDM signal generation."""
    fft_size: int = 64
    cp_ratio: float = 0.25  # Cyclic prefix ratio (e.g., 1/4 = 0.25)
    guard_band_ratio: float = 0.1  # Ratio of subcarriers with no data on each edge
    modulation: ModulationType = 'QPSK'

    @property
    def cp_length(self) -> int:
        """Cyclic prefix length in samples."""
        return int(self.fft_size * self.cp_ratio)

    @property
    def symbol_length(self) -> int:
        """Total OFDM symbol length including CP."""
        return self.fft_size + self.cp_length

    @property
    def num_guard_subcarriers(self) -> int:
        """Number of guard subcarriers on each edge."""
        return int(self.fft_size * self.guard_band_ratio / 2)

    @property
    def num_data_subcarriers(self) -> int:
        """
        Number of data subcarriers (only positive frequencies, excluding DC and Nyquist).
        For Hermitian symmetry, we only modulate positive frequency bins.
        """
        # Usable positive freq bins: 1 to N/2-1 (excluding DC=0 and Nyquist=N/2)
        usable_positive = self.fft_size // 2 - 1
        # Subtract guard bands
        return usable_positive - self.num_guard_subcarriers


def generate_qam_symbols(
    modulation: ModulationType,
    num_symbols: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random QAM symbols.

    Args:
        modulation: Modulation type ('QPSK', '16QAM', '64QAM', '256QAM')
        num_symbols: Number of symbols to generate
        seed: Random seed for reproducibility

    Returns:
        Complex array of QAM symbols with shape (num_symbols,)
    """
    if seed is not None:
        np.random.seed(seed)

    constellation = QAM_CONSTELLATIONS[modulation]
    indices = np.random.randint(0, len(constellation), size=num_symbols)
    return constellation[indices]


def generate_ofdm_symbol(
    data_symbols: np.ndarray,
    params: OFDMParams
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single OFDM symbol with Hermitian symmetry.

    The frequency-domain symbols are arranged with Hermitian symmetry so that
    the IFFT produces a real-valued time-domain signal:
    - X[0] = 0 (DC, set to zero for simplicity)
    - X[1:N/2] = data (positive frequencies)
    - X[N/2] = 0 (Nyquist, set to zero)
    - X[N/2+1:N] = conj(X[N/2-1:0:-1]) (negative frequencies, conjugate symmetric)

    Args:
        data_symbols: Complex QAM symbols for positive frequency bins
        params: OFDM parameters

    Returns:
        Tuple of (time_domain_signal, frequency_domain_symbols)
        - time_domain_signal: Real-valued signal with CP, shape (symbol_length,)
        - frequency_domain_symbols: Full frequency-domain representation, shape (fft_size,)
    """
    N = params.fft_size
    num_data = params.num_data_subcarriers
    num_guard = params.num_guard_subcarriers

    assert len(data_symbols) == num_data, \
        f"Expected {num_data} symbols, got {len(data_symbols)}"

    # Initialize frequency-domain representation
    X = np.zeros(N, dtype=np.complex128)

    # Place data symbols in positive frequency bins (after guard band, before Nyquist)
    # Positive freq bins to use: [1 + guard, ..., N/2 - 1]
    start_idx = 1 + num_guard
    end_idx = start_idx + num_data
    X[start_idx:end_idx] = data_symbols

    # Apply Hermitian symmetry for negative frequencies
    # X[N-k] = conj(X[k]) for k = 1 to N/2-1
    for k in range(1, N // 2):
        X[N - k] = np.conj(X[k])

    # IFFT to get time-domain signal (should be real due to Hermitian symmetry)
    x = np.fft.ifft(X)

    # Take real part (should be effectively real, imaginary part is numerical noise)
    x_real = np.real(x)

    # Add cyclic prefix
    cp_length = params.cp_length
    x_with_cp = np.concatenate([x_real[-cp_length:], x_real])

    return x_with_cp, X


def generate_ofdm_signal(
    num_ofdm_symbols: int,
    params: OFDMParams,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a complete OFDM signal with multiple symbols.

    Args:
        num_ofdm_symbols: Number of OFDM symbols to generate
        params: OFDM parameters
        seed: Random seed for reproducibility

    Returns:
        Tuple of:
        - time_signal: Real-valued time-domain signal, shape (num_ofdm_symbols * symbol_length,)
        - all_data_symbols: All QAM data symbols, shape (num_ofdm_symbols, num_data_subcarriers)
        - all_freq_symbols: All frequency-domain representations, shape (num_ofdm_symbols, fft_size)
    """
    if seed is not None:
        np.random.seed(seed)

    num_data = params.num_data_subcarriers

    all_time_symbols = []
    all_data_symbols = []
    all_freq_symbols = []

    for _ in range(num_ofdm_symbols):
        # Generate random QAM symbols for this OFDM symbol
        data_symbols = generate_qam_symbols(params.modulation, num_data)

        # Generate OFDM symbol
        time_symbol, freq_symbol = generate_ofdm_symbol(data_symbols, params)

        all_time_symbols.append(time_symbol)
        all_data_symbols.append(data_symbols)
        all_freq_symbols.append(freq_symbol)

    # Concatenate all time-domain symbols
    time_signal = np.concatenate(all_time_symbols)
    all_data_symbols = np.array(all_data_symbols)
    all_freq_symbols = np.array(all_freq_symbols)

    return time_signal, all_data_symbols, all_freq_symbols


def demodulate_ofdm(
    time_signal: Union[np.ndarray, torch.Tensor],
    params: OFDMParams,
    num_ofdm_symbols: Optional[int] = None
) -> Union[np.ndarray, torch.Tensor]:
    """
    Demodulate OFDM signal to extract QAM constellation points.

    Args:
        time_signal: Time-domain OFDM signal
        params: OFDM parameters used for generation
        num_ofdm_symbols: Number of OFDM symbols (inferred if not provided)

    Returns:
        Recovered QAM symbols, shape (num_ofdm_symbols, num_data_subcarriers)
    """
    is_torch = isinstance(time_signal, torch.Tensor)

    if is_torch:
        device = time_signal.device
        time_signal_np = time_signal.detach().cpu().numpy()
    else:
        time_signal_np = time_signal

    # Handle batched input
    if time_signal_np.ndim == 1:
        time_signal_np = time_signal_np[np.newaxis, :]
        was_unbatched = True
    else:
        was_unbatched = False

    batch_size = time_signal_np.shape[0]
    signal_len = time_signal_np.shape[1]

    symbol_length = params.symbol_length

    if num_ofdm_symbols is None:
        num_ofdm_symbols = signal_len // symbol_length

    num_data = params.num_data_subcarriers
    num_guard = params.num_guard_subcarriers
    N = params.fft_size
    cp_length = params.cp_length

    # Extract data indices
    start_idx = 1 + num_guard
    end_idx = start_idx + num_data

    all_symbols = []

    for b in range(batch_size):
        batch_symbols = []
        for i in range(num_ofdm_symbols):
            # Extract OFDM symbol (skip CP)
            start = i * symbol_length + cp_length
            end = start + N

            if end > signal_len:
                break

            x = time_signal_np[b, start:end]

            # FFT to frequency domain
            X = np.fft.fft(x)

            # Extract data symbols from positive frequency bins
            data_symbols = X[start_idx:end_idx]
            batch_symbols.append(data_symbols)

        all_symbols.append(np.array(batch_symbols))

    result = np.array(all_symbols)

    if was_unbatched:
        result = result[0]

    if is_torch:
        result = torch.from_numpy(result).to(device)

    return result


def demodulate_ofdm_torch(
    time_signal: torch.Tensor,
    params: OFDMParams,
    num_ofdm_symbols: Optional[int] = None
) -> torch.Tensor:
    """
    Demodulate OFDM signal using PyTorch operations (differentiable).

    Args:
        time_signal: Time-domain OFDM signal, shape (batch, signal_length) or (signal_length,)
        params: OFDM parameters used for generation
        num_ofdm_symbols: Number of OFDM symbols (inferred if not provided)

    Returns:
        Recovered QAM symbols, shape (batch, num_ofdm_symbols, num_data_subcarriers)
    """
    # Handle unbatched input
    if time_signal.ndim == 1:
        time_signal = time_signal.unsqueeze(0)
        was_unbatched = True
    else:
        was_unbatched = False

    batch_size = time_signal.shape[0]
    signal_len = time_signal.shape[1]

    symbol_length = params.symbol_length
    N = params.fft_size
    cp_length = params.cp_length

    if num_ofdm_symbols is None:
        num_ofdm_symbols = signal_len // symbol_length

    num_data = params.num_data_subcarriers
    num_guard = params.num_guard_subcarriers

    # Extract data indices
    start_idx = 1 + num_guard
    end_idx = start_idx + num_data

    all_symbols = []

    for i in range(num_ofdm_symbols):
        # Extract OFDM symbol (skip CP)
        start = i * symbol_length + cp_length
        end = start + N

        if end > signal_len:
            break

        x = time_signal[:, start:end]  # (batch, N)

        # FFT to frequency domain (torch.fft.fft works on real input)
        X = torch.fft.fft(x)  # (batch, N) complex

        # Extract data symbols from positive frequency bins
        data_symbols = X[:, start_idx:end_idx]  # (batch, num_data)
        all_symbols.append(data_symbols)

    # Stack all symbols: (num_ofdm_symbols, batch, num_data) -> (batch, num_ofdm_symbols, num_data)
    result = torch.stack(all_symbols, dim=1)

    if was_unbatched:
        result = result.squeeze(0)

    return result


# Utility functions for signal processing
def normalize_signal(signal: np.ndarray, target_power: float = 1.0) -> np.ndarray:
    """Normalize signal to target average power."""
    current_power = np.mean(signal ** 2)
    return signal * np.sqrt(target_power / current_power)


def apply_clipping(signal: np.ndarray, clip_ratio: float) -> np.ndarray:
    """
    Apply clipping to signal.

    Args:
        signal: Input signal
        clip_ratio: Clipping ratio relative to peak (e.g., 0.5 clips at half peak)

    Returns:
        Clipped signal
    """
    peak = np.max(np.abs(signal))
    clip_level = peak * clip_ratio
    return np.clip(signal, -clip_level, clip_level)


def get_papr(signal: np.ndarray) -> float:
    """Calculate Peak-to-Average Power Ratio in dB."""
    peak_power = np.max(signal ** 2)
    avg_power = np.mean(signal ** 2)
    return 10 * np.log10(peak_power / avg_power)
