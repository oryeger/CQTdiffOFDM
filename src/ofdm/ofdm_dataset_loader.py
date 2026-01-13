"""
Dataset loaders for OFDM signals (WAV format).

Compatible with the existing training pipeline structure (similar to MAESTRO loader).
Symbols are regenerated from seed stored in CSV for EVM calculation.
"""

import os
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Tuple, Dict

from src.ofdm.ofdm_generator import OFDMParams, generate_ofdm_signal


DEFAULT_METADATA_FILENAME = "ofdm_dataset.csv"


def get_metadata_file(dset_args) -> str:
    """Get metadata CSV file path."""
    csv_name = getattr(dset_args, "csv_name", DEFAULT_METADATA_FILENAME)
    return os.path.join(dset_args.path, csv_name)


def regenerate_symbols_from_row(row: pd.Series) -> Tuple[np.ndarray, OFDMParams]:
    """
    Regenerate QAM symbols from metadata row.

    Args:
        row: Pandas Series with metadata

    Returns:
        Tuple of (data_symbols, ofdm_params)
    """
    params = OFDMParams(
        fft_size=int(row['fft_size']),
        cp_ratio=float(row['cp_ratio']),
        guard_band_ratio=float(row['guard_band_ratio']),
        modulation=str(row['modulation'])
    )

    # Regenerate with same seed
    _, data_symbols, _ = generate_ofdm_signal(
        int(row['num_ofdm_symbols']),
        params,
        seed=int(row['seed'])
    )

    return data_symbols, params


class OFDMTrainDataset(IterableDataset):
    """
    Training dataset for OFDM signals (WAV files).

    Yields random segments from OFDM signal files.
    Compatible with the existing training pipeline.
    """

    def __init__(self, dset_args, fs=44100, seg_len=131072, seed=42):
        """
        Args:
            dset_args: Dataset arguments (hydra config)
            fs: Expected sample rate (44100 Hz)
            seg_len: Segment length for training
            seed: Random seed
        """
        super(OFDMTrainDataset).__init__()
        random.seed(seed)
        np.random.seed(seed)

        path = dset_args.path
        years = dset_args.years

        metadata_file = get_metadata_file(dset_args)
        metadata = pd.read_csv(metadata_file)

        # Filter by year and split
        metadata = metadata[metadata["year"].isin(years)]
        metadata = metadata[metadata["split"] == "train"]

        filelist = metadata["audio_filename"]
        filelist = filelist.map(lambda x: os.path.join(path, x), na_action='ignore')

        self.train_samples = filelist.to_list()

        # Optional: limit number of files
        if hasattr(dset_args, "max_files") and dset_args.max_files is not None:
            self.train_samples = self.train_samples[:int(dset_args.max_files)]

        self.seg_len = int(seg_len)
        self.fs = fs

    def __iter__(self):
        while True:
            # Random file selection
            num = random.randint(0, len(self.train_samples) - 1)
            file = self.train_samples[num]

            # Load WAV file
            data, samplerate = sf.read(file)

            if samplerate != self.fs:
                print(f"Warning: Sample rate mismatch: {samplerate} vs {self.fs}")

            # Convert stereo to mono (average channels)
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # If signal is shorter than seg_len, pad
            if len(data) < self.seg_len:
                data = np.pad(data, (0, self.seg_len - len(data)))

            # Get random segment
            if len(data) > self.seg_len:
                idx = np.random.randint(0, len(data) - self.seg_len)
                segment = data[idx:idx + self.seg_len]
            else:
                segment = data[:self.seg_len]

            segment = segment.astype('float32')
            yield segment


class OFDMTestDataset(Dataset):
    """
    Test/validation dataset for OFDM signals (WAV files).

    Returns full signals with regenerated QAM symbols for EVM computation.
    """

    def __init__(self, dset_args, fs=44100, seg_len=131072, split='test', return_symbols=True):
        """
        Args:
            dset_args: Dataset arguments (hydra config)
            fs: Expected sample rate (44100 Hz)
            seg_len: Segment length
            split: 'test' or 'validation'
            return_symbols: If True, return regenerated QAM symbols along with signal
        """
        path = dset_args.path

        if split == 'test':
            years = dset_args.years_test
        else:
            years = getattr(dset_args, 'years_val', dset_args.years_test)

        metadata_file = get_metadata_file(dset_args)
        metadata = pd.read_csv(metadata_file)

        # Filter by year and split
        metadata = metadata[metadata["year"].isin(years)]
        metadata = metadata[metadata["split"] == split]

        self.filelist = metadata["audio_filename"].map(
            lambda x: os.path.join(path, x), na_action='ignore'
        ).to_list()

        self.metadata = metadata.reset_index(drop=True)
        self.seg_len = int(seg_len)
        self.fs = fs
        self.return_symbols = return_symbols

        # Preload all data
        print(f"Loading {split} files...")
        self.signals = []
        self.symbols = []
        self.params = []

        for idx, f in enumerate(tqdm(self.filelist)):
            # Load WAV file
            data, samplerate = sf.read(f)

            if samplerate != self.fs:
                print(f"Warning: Sample rate mismatch: {samplerate} vs {self.fs}")

            # Convert stereo to mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Pad/truncate to seg_len
            if len(data) < self.seg_len:
                data = np.pad(data, (0, self.seg_len - len(data)))
            elif len(data) > self.seg_len:
                data = data[:self.seg_len]

            self.signals.append(data.astype('float32'))

            if self.return_symbols:
                # Regenerate symbols from seed
                row = self.metadata.iloc[idx]
                symbols, ofdm_params = regenerate_symbols_from_row(row)
                self.symbols.append(symbols)
                self.params.append({
                    'modulation': str(row['modulation']),
                    'fft_size': int(row['fft_size']),
                    'cp_ratio': float(row['cp_ratio']),
                    'guard_band_ratio': float(row['guard_band_ratio']),
                    'num_ofdm_symbols': int(row['num_ofdm_symbols']),
                    'seed': int(row['seed'])
                })

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        if self.return_symbols:
            return self.signals[idx], self.symbols[idx], self.params[idx]
        else:
            return self.signals[idx]


class OFDMTestDatasetWithMetadata(Dataset):
    """
    Test dataset that returns signals with all associated metadata.
    Loads files on-demand (not preloaded).
    """

    def __init__(self, dset_args, fs=44100, seg_len=131072, split='test'):
        path = dset_args.path

        if split == 'test':
            years = dset_args.years_test
        else:
            years = getattr(dset_args, 'years_val', dset_args.years_test)

        metadata_file = get_metadata_file(dset_args)
        metadata = pd.read_csv(metadata_file)

        metadata = metadata[metadata["year"].isin(years)]
        metadata = metadata[metadata["split"] == split]

        self.filelist = metadata["audio_filename"].map(
            lambda x: os.path.join(path, x), na_action='ignore'
        ).to_list()

        self.metadata_df = metadata.reset_index(drop=True)
        self.seg_len = int(seg_len)
        self.fs = fs
        self.path = path

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx) -> Dict:
        """Returns a dictionary with signal, symbols, and all parameters."""
        filepath = self.filelist[idx]
        row = self.metadata_df.iloc[idx]

        # Load WAV file
        data, samplerate = sf.read(filepath)

        # Convert stereo to mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Pad/truncate
        if len(data) < self.seg_len:
            data = np.pad(data, (0, self.seg_len - len(data)))
        elif len(data) > self.seg_len:
            data = data[:self.seg_len]

        # Regenerate symbols from seed
        symbols, ofdm_params = regenerate_symbols_from_row(row)

        return {
            'signal': data.astype('float32'),
            'symbols': symbols,
            'modulation': str(row['modulation']),
            'fft_size': int(row['fft_size']),
            'cp_ratio': float(row['cp_ratio']),
            'guard_band_ratio': float(row['guard_band_ratio']),
            'num_ofdm_symbols': int(row['num_ofdm_symbols']),
            'seed': int(row['seed']),
            'filename': os.path.basename(filepath)
        }


def collate_ofdm_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for OFDMTestDatasetWithMetadata.

    Handles batching of signals and symbols with different shapes.
    """
    signals = torch.stack([torch.from_numpy(item['signal']) for item in batch])

    # Symbols may have different shapes due to different num_ofdm_symbols
    # Keep as list
    symbols = [item['symbols'] for item in batch]

    # Collect parameters
    params = [{
        'modulation': item['modulation'],
        'fft_size': item['fft_size'],
        'cp_ratio': item['cp_ratio'],
        'guard_band_ratio': item['guard_band_ratio'],
        'num_ofdm_symbols': item['num_ofdm_symbols'],
        'seed': item['seed']
    } for item in batch]

    filenames = [item['filename'] for item in batch]

    return {
        'signals': signals,
        'symbols': symbols,
        'params': params,
        'filenames': filenames
    }
