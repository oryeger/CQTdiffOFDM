"""
Generate OFDM dataset for training, testing, and validation.

Creates directory structure matching MAESTRO format:
examples/ofdm_custom/
├── train/
│   └── ofdm_XXXXXX.wav  (stereo, 44100 Hz)
├── test/
│   └── ofdm_XXXXXX.wav
├── validation/
│   └── ofdm_XXXXXX.wav
└── ofdm_dataset.csv

The CSV contains all metadata including the random seed used to generate
each signal, allowing symbols to be regenerated for EVM calculation.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ofdm.ofdm_generator import OFDMParams, generate_ofdm_signal, normalize_signal, get_papr

# Constants matching MAESTRO dataset format
SAMPLE_RATE = 44100
NUM_CHANNELS = 2  # Stereo


def generate_dataset(
    output_dir: str,
    num_train: int = 10000,
    num_test: int = 1000,
    num_val: int = 500,
    modulations: list = None,
    fft_sizes: list = None,
    cp_ratios: list = None,
    guard_band_ratios: list = None,
    fill_ratio: tuple = (0.8, 1.0),  # Fill 80-100% of target length with OFDM signal
    target_signal_length: int = 131072,  # ~3 seconds at 44100 Hz
    seed: int = 42,
    target_power: float = 0.1  # Target RMS power for normalization
):
    """
    Generate OFDM dataset as WAV files (stereo, 44100 Hz).

    Args:
        output_dir: Output directory for dataset
        num_train: Number of training samples
        num_test: Number of test samples
        num_val: Number of validation samples
        modulations: List of modulation types to use
        fft_sizes: List of FFT sizes to use
        cp_ratios: List of CP ratios to use
        guard_band_ratios: List of guard band ratios to use
        fill_ratio: Tuple (min, max) ratio of target length to fill with OFDM signal
        target_signal_length: Target length in samples at 44100 Hz
        seed: Base random seed
        target_power: Target RMS power for signal normalization
    """
    np.random.seed(seed)

    # Default parameters
    if modulations is None:
        modulations = ['QPSK', '16QAM', '64QAM', '256QAM']
    if fft_sizes is None:
        fft_sizes = [64, 128, 256, 512]
    if cp_ratios is None:
        cp_ratios = [0.25, 0.125, 0.0625]  # 1/4, 1/8, 1/16
    if guard_band_ratios is None:
        guard_band_ratios = [0.1, 0.15, 0.2]

    # Create directories
    output_path = Path(output_dir)
    for split in ['train', 'test', 'validation']:
        (output_path / split).mkdir(parents=True, exist_ok=True)

    # Metadata list
    metadata = []

    # Assign "years" for compatibility with existing data loader
    # Train: 2003, Test: 2010, Validation: 2005
    year_map = {'train': 2003, 'test': 2010, 'validation': 2005}

    def generate_sample(sample_idx: int, split: str, base_seed: int):
        """Generate a single sample."""
        # Unique seed for this sample (stored in CSV for symbol regeneration)
        sample_seed = base_seed + sample_idx

        # Set seed for reproducibility
        np.random.seed(sample_seed)

        # Random parameters for this sample
        modulation = np.random.choice(modulations)
        fft_size = int(np.random.choice(fft_sizes))
        cp_ratio = float(np.random.choice(cp_ratios))
        guard_band_ratio = float(np.random.choice(guard_band_ratios))

        params = OFDMParams(
            fft_size=fft_size,
            cp_ratio=cp_ratio,
            guard_band_ratio=guard_band_ratio,
            modulation=modulation
        )

        # Calculate how many OFDM symbols to fill the target length
        symbol_length = params.symbol_length

        # Calculate min and max number of symbols based on fill_ratio
        min_symbols = max(1, int(target_signal_length * fill_ratio[0] / symbol_length))
        max_symbols = int(target_signal_length * fill_ratio[1] / symbol_length)

        if max_symbols < min_symbols:
            max_symbols = min_symbols

        num_ofdm_symbols = np.random.randint(min_symbols, max_symbols + 1)

        # Generate OFDM signal (seed is already set above)
        time_signal, data_symbols, freq_symbols = generate_ofdm_signal(
            num_ofdm_symbols, params, seed=sample_seed
        )

        # Normalize signal to target power
        time_signal = normalize_signal(time_signal, target_power=target_power)

        # Pad or truncate to target length
        if len(time_signal) < target_signal_length:
            # Pad with zeros
            time_signal = np.pad(time_signal, (0, target_signal_length - len(time_signal)))
        elif len(time_signal) > target_signal_length:
            # Truncate
            time_signal = time_signal[:target_signal_length]

        # Calculate PAPR
        papr = get_papr(time_signal)

        # Convert to stereo (duplicate mono to both channels)
        stereo_signal = np.column_stack([time_signal, time_signal])

        # Save as WAV file
        filename = f"ofdm_{sample_idx:06d}.wav"
        filepath = output_path / split / filename

        # Write WAV file (soundfile handles float normalization)
        sf.write(filepath, stereo_signal.astype(np.float32), SAMPLE_RATE)

        # Return metadata (includes seed for symbol regeneration)
        return {
            'audio_filename': f"{split}/{filename}",
            'split': split,
            'year': year_map[split],
            'modulation': modulation,
            'fft_size': fft_size,
            'cp_ratio': cp_ratio,
            'guard_band_ratio': guard_band_ratio,
            'num_ofdm_symbols': num_ofdm_symbols,
            'signal_length': target_signal_length,
            'sample_rate': SAMPLE_RATE,
            'num_channels': NUM_CHANNELS,
            'seed': sample_seed,  # For symbol regeneration
            'papr_db': papr
        }

    # Generate training samples
    print("Generating training samples...")
    for i in tqdm(range(num_train)):
        meta = generate_sample(i, 'train', seed)
        metadata.append(meta)

    # Generate test samples (offset seed to ensure different samples)
    print("Generating test samples...")
    for i in tqdm(range(num_test)):
        meta = generate_sample(num_train + i, 'test', seed + 1000000)
        metadata.append(meta)

    # Generate validation samples
    print("Generating validation samples...")
    for i in tqdm(range(num_val)):
        meta = generate_sample(num_train + num_test + i, 'validation', seed + 2000000)
        metadata.append(meta)

    # Save metadata CSV
    df = pd.DataFrame(metadata)
    csv_path = output_path / 'ofdm_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved metadata to {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("Dataset Summary:")
    print("=" * 60)
    print(f"  Output directory: {output_path}")
    print(f"  Training samples: {num_train}")
    print(f"  Test samples: {num_test}")
    print(f"  Validation samples: {num_val}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Channels: {NUM_CHANNELS} (stereo)")
    print(f"  Signal length: {target_signal_length} samples ({target_signal_length/SAMPLE_RATE:.2f} sec)")
    print(f"  Fill ratio: {fill_ratio[0]*100:.0f}%-{fill_ratio[1]*100:.0f}% of signal length")
    print(f"  Modulations: {modulations}")
    print(f"  FFT sizes: {fft_sizes}")
    print(f"  CP ratios: {cp_ratios}")
    print(f"  Guard band ratios: {guard_band_ratios}")
    print("=" * 60)

    return df


def regenerate_symbols_from_metadata(metadata_row: dict):
    """
    Regenerate QAM symbols from metadata (for EVM calculation).

    Args:
        metadata_row: Dictionary with 'seed', 'modulation', 'fft_size',
                      'cp_ratio', 'guard_band_ratio', 'num_ofdm_symbols'

    Returns:
        Tuple of (data_symbols, ofdm_params)
    """
    from src.ofdm.ofdm_generator import generate_ofdm_signal

    params = OFDMParams(
        fft_size=int(metadata_row['fft_size']),
        cp_ratio=float(metadata_row['cp_ratio']),
        guard_band_ratio=float(metadata_row['guard_band_ratio']),
        modulation=str(metadata_row['modulation'])
    )

    # Regenerate with same seed
    _, data_symbols, _ = generate_ofdm_signal(
        int(metadata_row['num_ofdm_symbols']),
        params,
        seed=int(metadata_row['seed'])
    )

    return data_symbols, params


def main():
    parser = argparse.ArgumentParser(description='Generate OFDM dataset (WAV files)')
    parser.add_argument('--output_dir', type=str, default='examples/ofdm_custom',
                        help='Output directory for dataset')
    parser.add_argument('--num_train', type=int, default=10000,
                        help='Number of training samples')
    parser.add_argument('--num_test', type=int, default=1000,
                        help='Number of test samples')
    parser.add_argument('--num_val', type=int, default=500,
                        help='Number of validation samples')
    parser.add_argument('--signal_length', type=int, default=131072,
                        help='Target signal length in samples (~3 sec at 44100 Hz)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')
    parser.add_argument('--target_power', type=float, default=0.1,
                        help='Target RMS power for normalization')

    args = parser.parse_args()

    # Convert to absolute path if relative
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)

    generate_dataset(
        output_dir=args.output_dir,
        num_train=args.num_train,
        num_test=args.num_test,
        num_val=args.num_val,
        target_signal_length=args.signal_length,
        seed=args.seed,
        target_power=args.target_power
    )


if __name__ == '__main__':
    main()
