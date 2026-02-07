"""
Dataset for OFDM Declipping Training.

Generates complex IQ OFDM signals with on-the-fly MAGNITUDE-BASED clipping.

IMPORTANT: Clipping is applied based on the complex magnitude |x| = sqrt(I² + Q²),
NOT independently on I and Q. This matches real RF/baseband clipping behavior where
the limiter operates on the signal envelope.

When |x| >= A (clip threshold):
    - The signal is scaled to have magnitude A while preserving phase
    - x_clipped = A * x / |x|  (same direction, limited magnitude)

The mask indicates where clipping occurred: m = 1 where |x_clean| >= A

Returns:
- clean_signal: Clean complex IQ, shape (2, T) [I, Q channels]
- clipped_signal: Clipped observation, shape (2, T)
- mask: Clipping mask (1 where |x| >= A), shape (1, T) based on MAGNITUDE
- symbols: QAM data symbols for EVM computation
- ofdm_params: OFDM parameters
- clip_level: Scalar clip threshold A
"""

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union
import random

from src.ofdm.ofdm_generator import (
    OFDMParams,
    generate_ofdm_signal,
    generate_qam_symbols,
    QAM_CONSTELLATIONS,
)


@dataclass
class DeclipConfig:
    """Configuration for declipping dataset."""
    # Signal parameters
    signal_length: int = 8192  # Time samples
    num_ofdm_symbols: int = 100  # Number of OFDM symbols per signal
    
    # OFDM parameters
    fft_size: int = 64
    cp_ratio: float = 0.25
    guard_band_ratio: float = 0.1
    modulation: str = '16QAM'
    
    # Clipping parameters - LESS AGGRESSIVE (only clip top ~10-20% of peaks)
    clip_ratio_min: float = 0.75  # Min clip ratio (mild clipping)
    clip_ratio_max: float = 0.92  # Max clip ratio (very mild)
    
    # Normalization
    normalize_rms: bool = True  # Normalize to unit RMS
    
    # Randomization
    random_ofdm_params: bool = False  # Randomize FFT size, modulation
    random_fft_sizes: Tuple[int, ...] = (64, 128, 256)
    random_modulations: Tuple[str, ...] = ('QPSK', '16QAM', '64QAM')


class OFDMDeclipDataset(IterableDataset):
    """
    Iterable dataset for OFDM declipping training.
    
    Generates TRUE COMPLEX baseband IQ signals with on-the-fly clipping.
    
    The signal is complex baseband OFDM where BOTH I and Q channels contain 
    actual data (not Hermitian-symmetric). This matches real wireless systems
    where the baseband signal is complex-valued.
    """
    
    def __init__(
        self,
        config: Optional[DeclipConfig] = None,
        seed: int = 42,
        complex_mode: str = 'true_complex',  # 'true_complex', 'real_as_iq', or 'analytic'
    ):
        """
        Args:
            config: Dataset configuration
            seed: Random seed
            complex_mode: How to create complex representation
                - 'true_complex': Generate true complex baseband OFDM (RECOMMENDED)
                - 'real_as_iq': I=signal, Q=zeros (only for testing)
                - 'analytic': I=signal, Q=Hilbert(signal)
        """
        super().__init__()
        self.config = config or DeclipConfig()
        self.complex_mode = complex_mode
        
        random.seed(seed)
        np.random.seed(seed)
    
    def _get_ofdm_params(self) -> OFDMParams:
        """Get OFDM parameters (potentially randomized)."""
        cfg = self.config
        
        if cfg.random_ofdm_params:
            fft_size = random.choice(cfg.random_fft_sizes)
            modulation = random.choice(cfg.random_modulations)
        else:
            fft_size = cfg.fft_size
            modulation = cfg.modulation
        
        return OFDMParams(
            fft_size=fft_size,
            cp_ratio=cfg.cp_ratio,
            guard_band_ratio=cfg.guard_band_ratio,
            modulation=modulation,
        )
    
    def _generate_signal(self) -> Tuple[np.ndarray, np.ndarray, OFDMParams]:
        """Generate a clean OFDM signal (real-valued via Hermitian symmetry)."""
        params = self._get_ofdm_params()
        
        # Calculate number of symbols to fill signal_length
        symbol_length = params.symbol_length
        num_symbols = self.config.signal_length // symbol_length
        
        # Generate signal
        time_signal, data_symbols, _ = generate_ofdm_signal(
            num_symbols,
            params,
            seed=None  # Random each time
        )
        
        # Truncate or pad to exact length
        if len(time_signal) > self.config.signal_length:
            time_signal = time_signal[:self.config.signal_length]
        elif len(time_signal) < self.config.signal_length:
            time_signal = np.pad(
                time_signal,
                (0, self.config.signal_length - len(time_signal))
            )
        
        return time_signal, data_symbols, params
    
    def _generate_complex_baseband_ofdm(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Generate TRUE COMPLEX baseband OFDM signal.
        
        This generates a complex-valued signal where BOTH I and Q channels
        contain actual data. This is what real wireless systems use at baseband.
        
        Key difference from Hermitian-symmetric version:
        - All subcarriers (not just positive frequencies) can carry independent data
        - No Hermitian symmetry constraint
        - Time-domain signal is truly complex (I ≠ 0 AND Q ≠ 0)
        
        Returns:
            signal_iq: Complex signal as (2, T) array [I, Q]
            data_symbols: QAM symbols used
            metadata: Signal parameters
        """
        cfg = self.config
        
        # Get parameters
        if cfg.random_ofdm_params:
            fft_size = random.choice(cfg.random_fft_sizes)
            modulation = random.choice(cfg.random_modulations)
        else:
            fft_size = cfg.fft_size
            modulation = cfg.modulation
        
        cp_length = int(fft_size * cfg.cp_ratio)
        symbol_length = fft_size + cp_length
        num_guard = int(fft_size * cfg.guard_band_ratio / 2)
        
        # Calculate number of OFDM symbols needed
        num_symbols = max(1, self.config.signal_length // symbol_length)
        
        # Data subcarrier indices (exclude guard bands and DC)
        data_indices = []
        for k in range(fft_size):
            # Skip guard bands
            if k < num_guard or k >= fft_size - num_guard:
                continue
            # Skip DC (optional, can be included in some systems)
            if k == fft_size // 2:
                continue
            data_indices.append(k)
        
        num_data_subcarriers = len(data_indices)
        
        # Generate QAM constellation
        constellation = generate_qam_symbols(modulation, 1)  # Get constellation
        constellation = QAM_CONSTELLATIONS.get(modulation, QAM_CONSTELLATIONS['QPSK'])
        
        # Generate random QAM symbols for all OFDM symbols
        total_data_symbols = num_data_subcarriers * num_symbols
        symbol_indices = np.random.randint(0, len(constellation), total_data_symbols)
        data_symbols = constellation[symbol_indices].reshape(num_symbols, num_data_subcarriers)
        
        # Build frequency-domain OFDM symbols (complex, NO Hermitian symmetry)
        freq_symbols = np.zeros((num_symbols, fft_size), dtype=np.complex128)
        freq_symbols[:, data_indices] = data_symbols
        
        # IFFT to time domain - result is COMPLEX (not real!)
        time_symbols = np.fft.ifft(freq_symbols, axis=1) * np.sqrt(fft_size)
        
        # Add cyclic prefix
        if cp_length > 0:
            cp = time_symbols[:, -cp_length:]
            time_symbols_cp = np.concatenate([cp, time_symbols], axis=1)
        else:
            time_symbols_cp = time_symbols
        
        # Concatenate all symbols into one signal
        signal_complex = time_symbols_cp.flatten()
        
        # Normalize to unit variance (using complex power)
        signal_power = np.mean(np.abs(signal_complex) ** 2)
        signal_complex = signal_complex / np.sqrt(signal_power)
        
        # Pad or truncate to target length
        if len(signal_complex) < self.config.signal_length:
            signal_complex = np.pad(signal_complex, (0, self.config.signal_length - len(signal_complex)))
        else:
            signal_complex = signal_complex[:self.config.signal_length]
        
        # Convert to 2-channel IQ: shape (2, T)
        signal_iq = np.stack([
            signal_complex.real.astype(np.float32),
            signal_complex.imag.astype(np.float32)
        ], axis=0)
        
        # Create OFDMParams-like metadata for frequency loss computation
        metadata = {
            'fft_size': fft_size,
            'cp_length': cp_length,
            'num_symbols': num_symbols,
            'num_data_subcarriers': num_data_subcarriers,
            'data_indices': data_indices,
            'modulation': modulation,
            'guard_band_ratio': cfg.guard_band_ratio,
        }
        
        return signal_iq, data_symbols, metadata
    
    def _to_complex_iq(self, signal: np.ndarray) -> np.ndarray:
        """
        Convert real signal to 2-channel IQ representation.
        
        Returns: (2, T) array with [I, Q] channels
        """
        if self.complex_mode == 'real_as_iq':
            # Simple: I = signal, Q = 0
            return np.stack([signal, np.zeros_like(signal)], axis=0)
        
        elif self.complex_mode == 'analytic':
            # Analytic signal: I = signal, Q = Hilbert(signal)
            from scipy.signal import hilbert
            analytic = hilbert(signal)
            return np.stack([np.real(analytic), np.imag(analytic)], axis=0)
        
        elif self.complex_mode == 'true_complex':
            # This mode uses _generate_complex_baseband_ofdm directly
            # This method shouldn't be called for true_complex mode
            raise RuntimeError("_to_complex_iq shouldn't be called for 'true_complex' mode")
        
        else:
            raise ValueError(f"Unknown complex_mode: {self.complex_mode}")
    
    def _apply_clipping(
        self,
        signal: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Apply MAGNITUDE-BASED clipping to complex IQ signal.
        
        Clipping is applied based on magnitude: if |x| = sqrt(I² + Q²) >= A,
        then scale the vector to have magnitude A while preserving phase:
            x_clipped = A * x / |x|
        
        This is the correct model for RF/baseband clipping where the 
        limiter operates on the envelope, not on I and Q independently.
        
        Args:
            signal: Clean signal, shape (2, T) with [I, Q] channels
        
        Returns:
            clipped: Clipped signal, shape (2, T)
            mask: Clipping mask (1 where |x| >= A), shape (1, T)
            clip_level: Clip threshold A
        """
        cfg = self.config
        
        # Random clip ratio
        clip_ratio = np.random.uniform(cfg.clip_ratio_min, cfg.clip_ratio_max)
        
        # Compute magnitude: |x| = sqrt(I² + Q²)
        I = signal[0]
        Q = signal[1]
        magnitude = np.sqrt(I ** 2 + Q ** 2)
        
        # Clip level based on peak magnitude
        peak = np.max(magnitude)
        clip_level = peak * clip_ratio
        
        # Create mask: m = 1 where |x| >= A (BEFORE clipping)
        is_clipped = magnitude >= clip_level
        mask = is_clipped.astype(np.float32)[np.newaxis, :]  # (1, T)
        
        # Apply MAGNITUDE clipping (preserve phase, limit magnitude)
        # x_clipped = x * min(1, A / |x|)
        # This scales the vector (I, Q) to have magnitude at most A
        scale = np.ones_like(magnitude)
        scale[is_clipped] = clip_level / (magnitude[is_clipped] + 1e-10)
        
        clipped = np.stack([
            I * scale,
            Q * scale
        ], axis=0)
        
        return clipped, mask, clip_level
    
    def _normalize(
        self,
        clean: np.ndarray,
        clipped: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Normalize signals to unit RMS.
        
        Returns:
            Normalized clean, normalized clipped, scale factor
        """
        # Compute RMS of clean signal
        rms = np.sqrt(np.mean(clean ** 2))
        
        if rms < 1e-8:
            return clean, clipped, 1.0
        
        return clean / rms, clipped / rms, rms
    
    def __iter__(self):
        while True:
            if self.complex_mode == 'true_complex':
                # Generate TRUE COMPLEX baseband OFDM directly
                clean_iq, data_symbols, ofdm_metadata = self._generate_complex_baseband_ofdm()
                
                # Apply clipping
                clipped_iq, mask, clip_level = self._apply_clipping(clean_iq)
                
                # Normalize
                if self.config.normalize_rms:
                    clean_iq, clipped_iq, scale = self._normalize(clean_iq, clipped_iq)
                    clip_level = clip_level / scale
                
                # Convert to tensors
                clean = torch.from_numpy(clean_iq.astype(np.float32))
                clipped = torch.from_numpy(clipped_iq.astype(np.float32))
                mask = torch.from_numpy(mask.astype(np.float32))
                clip_level_tensor = torch.tensor([clip_level], dtype=torch.float32)
                
                yield (clean, clipped, mask, data_symbols, ofdm_metadata, clip_level_tensor)
            else:
                # Legacy mode: Generate Hermitian-symmetric real signal, then convert
                time_signal, data_symbols, ofdm_params = self._generate_signal()
                
                # Convert to IQ representation
                clean_iq = self._to_complex_iq(time_signal)
                
                # Apply clipping
                clipped_iq, mask, clip_level = self._apply_clipping(clean_iq)
                
                # Normalize
                if self.config.normalize_rms:
                    clean_iq, clipped_iq, scale = self._normalize(clean_iq, clipped_iq)
                    clip_level = clip_level / scale
                
                # Convert to tensors
                clean = torch.from_numpy(clean_iq.astype(np.float32))
                clipped = torch.from_numpy(clipped_iq.astype(np.float32))
                mask = torch.from_numpy(mask.astype(np.float32))
                clip_level_tensor = torch.tensor([clip_level], dtype=torch.float32)
                
                yield (clean, clipped, mask, data_symbols, ofdm_params, clip_level_tensor)


class OFDMDeclipDatasetFixed(Dataset):
    """
    Fixed-size dataset (non-iterable) for validation/testing.
    
    Pre-generates a set of signals for consistent evaluation.
    """
    
    def __init__(
        self,
        config: Optional[DeclipConfig] = None,
        num_samples: int = 100,
        seed: int = 42,
        complex_mode: str = 'true_complex',  # Default to true complex signals
    ):
        super().__init__()
        self.config = config or DeclipConfig()
        self.num_samples = num_samples
        self.complex_mode = complex_mode
        
        # Pre-generate all samples
        self.samples = []
        self._generate_all(seed)
    
    def _generate_all(self, seed: int):
        """Pre-generate all samples."""
        np.random.seed(seed)
        random.seed(seed)
        
        iterable_ds = OFDMDeclipDataset(
            config=self.config,
            seed=seed,
            complex_mode=self.complex_mode,
        )
        
        iterator = iter(iterable_ds)
        for _ in range(self.num_samples):
            self.samples.append(next(iterator))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_declip_batch(batch: List) -> Tuple:
    """
    Custom collate function for declipping dataset.
    
    Handles varying OFDM parameters in batch.
    """
    clean_list, clipped_list, mask_list, symbols_list, params_list, clip_levels = zip(*batch)
    
    # Stack tensors
    clean = torch.stack(clean_list)
    clipped = torch.stack(clipped_list)
    mask = torch.stack(mask_list)
    clip_level = torch.stack(clip_levels)
    
    # For symbols and params, just take first (they may vary)
    # In practice, use same params for whole batch or handle individually
    symbols = symbols_list[0]
    params = params_list[0]
    
    return clean, clipped, mask, symbols, params, clip_level


def create_declip_dataloader(
    config: Optional[DeclipConfig] = None,
    batch_size: int = 8,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Create DataLoader for declipping training."""
    dataset = OFDMDeclipDataset(config, seed=seed)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_declip_batch,
        num_workers=num_workers,
    )


def create_test_dataloader(
    config: Optional[DeclipConfig] = None,
    num_samples: int = 100,
    batch_size: int = 1,
    seed: int = 12345,
) -> DataLoader:
    """Create DataLoader for testing/validation."""
    dataset = OFDMDeclipDatasetFixed(
        config,
        num_samples=num_samples,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_declip_batch,
        shuffle=False,
    )


# Utility functions

def compute_clip_statistics(mask: torch.Tensor) -> dict:
    """Compute clipping statistics from mask."""
    total_samples = mask.numel()
    clipped_samples = mask.sum().item()
    
    return {
        'clip_ratio': clipped_samples / total_samples,
        'clipped_samples': int(clipped_samples),
        'total_samples': total_samples,
    }


def compute_sdr(clean: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute Signal-to-Distortion Ratio in dB.
    
    SDR = 10 * log10(||clean||² / ||clean - recon||²)
    """
    clean_power = torch.sum(clean ** 2)
    error_power = torch.sum((clean - reconstructed) ** 2)
    
    return 10 * torch.log10(clean_power / (error_power + 1e-10)).item()


# Testing
if __name__ == "__main__":
    print("Testing OFDMDeclipDataset...")
    
    config = DeclipConfig(
        signal_length=8192,
        fft_size=64,
        modulation='16QAM',
        clip_ratio_min=0.4,
        clip_ratio_max=0.6,
    )
    
    dataloader = create_declip_dataloader(config, batch_size=4)
    
    batch = next(iter(dataloader))
    clean, clipped, mask, symbols, params, clip_level = batch
    
    print(f"Clean shape: {clean.shape}")
    print(f"Clipped shape: {clipped.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Symbols shape: {symbols.shape if symbols is not None else None}")
    print(f"Clip level: {clip_level}")
    print(f"OFDM params: FFT={params.fft_size}, mod={params.modulation}")
    
    stats = compute_clip_statistics(mask)
    print(f"Clipping stats: {stats}")
    
    print("✓ Dataset test passed!")
