"""
Complex Baseband OFDM Signal Generator for Diffusion Training.

Generates clean OFDM signals with randomized parameters for training
an unconditional diffusion prior.

Signal representation: 2 channels [Re(x), Im(x)]
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import random


@dataclass
class OFDMConfig:
    """OFDM configuration with randomization ranges."""
    # FFT sizes to randomly choose from
    fft_sizes: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    
    # Modulation orders
    modulations: List[str] = field(default_factory=lambda: ["QPSK", "16QAM", "64QAM"])
    
    # CP ratio range (fraction of FFT size)
    cp_ratio_range: Tuple[float, float] = (0.0625, 0.25)  # 1/16 to 1/4
    
    # Guard band ratio range (fraction of subcarriers nulled on edges)
    guard_band_range: Tuple[float, float] = (0.05, 0.15)
    
    # Number of OFDM symbols range
    num_symbols_range: Tuple[int, int] = (10, 50)
    
    # Oversampling factor (for analog-like waveform)
    oversampling: int = 1
    
    # Pilot density (fraction of data subcarriers used as pilots)
    pilot_density_range: Tuple[float, float] = (0.0, 0.1)
    
    # Windowing (raised cosine rolloff samples)
    window_rolloff_range: Tuple[int, int] = (0, 8)


# QAM constellation mappings
QAM_CONSTELLATIONS = {
    "QPSK": np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2),
    "16QAM": None,  # Generated below
    "64QAM": None,  # Generated below
}

def _generate_qam(M: int) -> np.ndarray:
    """Generate normalized M-QAM constellation."""
    sqrt_M = int(np.sqrt(M))
    points = []
    for i in range(sqrt_M):
        for j in range(sqrt_M):
            real = 2*i - sqrt_M + 1
            imag = 2*j - sqrt_M + 1
            points.append(real + 1j*imag)
    constellation = np.array(points)
    # Normalize to unit average power
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation)**2))
    return constellation

QAM_CONSTELLATIONS["16QAM"] = _generate_qam(16)
QAM_CONSTELLATIONS["64QAM"] = _generate_qam(64)


def generate_qam_symbols(
    num_symbols: int,
    modulation: str,
    seed: Optional[int] = None
) -> np.ndarray:
    """Generate random QAM symbols."""
    if seed is not None:
        np.random.seed(seed)
    
    constellation = QAM_CONSTELLATIONS[modulation]
    indices = np.random.randint(0, len(constellation), num_symbols)
    return constellation[indices]


def generate_ofdm_signal_complex(
    config: OFDMConfig,
    target_length: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, dict]:
    """
    Generate a complex baseband OFDM signal with randomized parameters.
    
    Args:
        config: OFDM configuration with randomization ranges
        target_length: Target signal length in samples
        seed: Random seed for reproducibility
        
    Returns:
        signal: Complex baseband signal, shape (target_length,)
        metadata: Dictionary with all OFDM parameters used
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    # Randomly select parameters
    fft_size = random.choice(config.fft_sizes)
    modulation = random.choice(config.modulations)
    cp_ratio = np.random.uniform(*config.cp_ratio_range)
    guard_band_ratio = np.random.uniform(*config.guard_band_range)
    pilot_density = np.random.uniform(*config.pilot_density_range)
    window_rolloff = np.random.randint(*config.window_rolloff_range) if config.window_rolloff_range[1] > 0 else 0
    
    # Derived parameters
    cp_length = int(fft_size * cp_ratio)
    symbol_length = fft_size + cp_length
    num_guard = int(fft_size * guard_band_ratio / 2)
    
    # Calculate number of symbols to fill target length
    num_symbols = max(1, target_length // symbol_length)
    
    # Subcarrier allocation
    num_data_subcarriers = fft_size - 2 * num_guard - 1  # -1 for DC null
    num_pilots = int(num_data_subcarriers * pilot_density)
    num_data = num_data_subcarriers - num_pilots
    
    # Create subcarrier mask
    subcarrier_mask = np.zeros(fft_size, dtype=bool)
    data_indices = []
    pilot_indices = []
    
    for k in range(fft_size):
        # Skip guard bands and DC
        if k < num_guard or k >= fft_size - num_guard:
            continue
        if k == fft_size // 2:  # DC null
            continue
        data_indices.append(k)
    
    # Randomly select some as pilots
    if num_pilots > 0 and len(data_indices) > num_pilots:
        pilot_indices = random.sample(data_indices, num_pilots)
        data_indices = [k for k in data_indices if k not in pilot_indices]
    
    subcarrier_mask[data_indices] = True
    
    # Generate data symbols
    total_data_symbols = len(data_indices) * num_symbols
    data_symbols = generate_qam_symbols(total_data_symbols, modulation, seed)
    data_symbols = data_symbols.reshape(num_symbols, len(data_indices))
    
    # Generate pilot symbols (BPSK for simplicity)
    if num_pilots > 0:
        pilot_symbols = np.ones((num_symbols, num_pilots)) * (1 + 0j)
    
    # Build frequency-domain OFDM symbols
    freq_symbols = np.zeros((num_symbols, fft_size), dtype=complex)
    freq_symbols[:, data_indices] = data_symbols
    if num_pilots > 0:
        freq_symbols[:, pilot_indices] = pilot_symbols
    
    # IFFT to get time domain
    time_symbols = np.fft.ifft(freq_symbols, axis=1) * np.sqrt(fft_size)
    
    # Add cyclic prefix
    if cp_length > 0:
        cp = time_symbols[:, -cp_length:]
        time_symbols_cp = np.concatenate([cp, time_symbols], axis=1)
    else:
        time_symbols_cp = time_symbols
    
    # Apply windowing (raised cosine)
    if window_rolloff > 0:
        window = np.ones(symbol_length)
        rolloff = np.sin(np.linspace(0, np.pi/2, window_rolloff))**2
        window[:window_rolloff] = rolloff
        window[-window_rolloff:] = rolloff[::-1]
        time_symbols_cp = time_symbols_cp * window
    
    # Concatenate symbols
    signal = time_symbols_cp.flatten()
    
    # Normalize to unit variance
    signal = signal / np.std(signal)
    
    # Pad or truncate to target length
    if len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))
    else:
        signal = signal[:target_length]
    
    metadata = {
        'fft_size': fft_size,
        'modulation': modulation,
        'cp_ratio': cp_ratio,
        'cp_length': cp_length,
        'guard_band_ratio': guard_band_ratio,
        'num_guard': num_guard,
        'num_symbols': num_symbols,
        'num_data_subcarriers': len(data_indices),
        'num_pilots': num_pilots,
        'data_indices': data_indices,
        'pilot_indices': pilot_indices,
        'data_symbols': data_symbols,
        'window_rolloff': window_rolloff,
        'seed': seed,
    }
    
    return signal, metadata


def complex_to_2channel(signal: np.ndarray) -> np.ndarray:
    """Convert complex signal to 2-channel real representation."""
    return np.stack([signal.real, signal.imag], axis=0).astype(np.float32)


def channel_2_to_complex(signal: np.ndarray) -> np.ndarray:
    """Convert 2-channel representation back to complex."""
    return signal[0] + 1j * signal[1]


class OFDMComplexDataset(torch.utils.data.IterableDataset):
    """
    Training dataset for complex baseband OFDM signals.
    
    Yields 2-channel tensors [Re(x), Im(x)] with randomized OFDM parameters.
    """
    
    def __init__(
        self,
        signal_length: int = 8192,
        config: Optional[OFDMConfig] = None
    ):
        """
        Args:
            signal_length: Target signal length in samples
            config: OFDM configuration (uses defaults if None)
        """
        super().__init__()
        self.signal_length = signal_length
        self.config = config or OFDMConfig()
    
    def __iter__(self):
        while True:
            # Generate random seed for this sample
            seed = np.random.randint(0, 2**31)
            
            # Generate complex OFDM signal
            signal, metadata = generate_ofdm_signal_complex(
                self.config, self.signal_length, seed
            )
            
            # Convert to 2-channel representation
            signal_2ch = complex_to_2channel(signal)
            
            yield signal_2ch


class OFDMComplexDatasetWithMetadata(torch.utils.data.IterableDataset):
    """
    Training dataset that also returns metadata for EVM-aware training.
    """
    
    def __init__(
        self,
        signal_length: int = 8192,
        config: Optional[OFDMConfig] = None
    ):
        super().__init__()
        self.signal_length = signal_length
        self.config = config or OFDMConfig()
    
    def __iter__(self):
        while True:
            seed = np.random.randint(0, 2**31)
            signal, metadata = generate_ofdm_signal_complex(
                self.config, self.signal_length, seed
            )
            signal_2ch = complex_to_2channel(signal)
            
            yield signal_2ch, metadata


def demodulate_ofdm_complex(
    signal_2ch: np.ndarray,
    metadata: dict
) -> np.ndarray:
    """
    Demodulate OFDM signal to recover QAM symbols.
    
    Args:
        signal_2ch: 2-channel signal [Re, Im], shape (2, T)
        metadata: OFDM parameters from generation
        
    Returns:
        recovered_symbols: Complex QAM symbols, shape (num_symbols, num_data_subcarriers)
    """
    # Convert back to complex
    signal = signal_2ch[0] + 1j * signal_2ch[1]
    
    fft_size = metadata['fft_size']
    cp_length = metadata['cp_length']
    num_symbols = metadata['num_symbols']
    data_indices = metadata['data_indices']
    symbol_length = fft_size + cp_length
    
    recovered_symbols = []
    
    for sym_idx in range(num_symbols):
        start = sym_idx * symbol_length + cp_length  # Skip CP
        end = start + fft_size
        
        if end > len(signal):
            break
        
        # Extract symbol and FFT
        time_symbol = signal[start:end]
        freq_symbol = np.fft.fft(time_symbol) / np.sqrt(fft_size)
        
        # Extract data subcarriers
        data_symbols = freq_symbol[data_indices]
        recovered_symbols.append(data_symbols)
    
    return np.array(recovered_symbols)


def compute_evm_complex(
    reference_symbols: np.ndarray,
    recovered_symbols: np.ndarray
) -> float:
    """
    Compute EVM between reference and recovered symbols.
    
    Args:
        reference_symbols: Original QAM symbols
        recovered_symbols: Demodulated symbols
        
    Returns:
        EVM as percentage
    """
    error = recovered_symbols - reference_symbols
    error_power = np.mean(np.abs(error)**2)
    ref_power = np.mean(np.abs(reference_symbols)**2)
    
    evm = np.sqrt(error_power / ref_power) * 100
    return evm
