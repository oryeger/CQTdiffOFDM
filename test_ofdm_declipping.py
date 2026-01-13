"""
Test OFDM declipping after training.

Usage:
    python test_ofdm_declipping.py model_dir=experiments/ofdm inference.checkpoint=weights-10000.pt
"""
import os
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf

from src.experimenters.exp_ofdm_declipping import Exp_OFDM_Declipping
from src.ofdm.ofdm_generator import OFDMParams
from src.ofdm.ofdm_dataset_loader import OFDMTestDatasetWithMetadata


def run(args):
    # Prepare args
    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))

    print(f"Loading model from: {args.model_dir}")
    print(f"Checkpoint: {args.inference.checkpoint}")

    # Create experimenter
    experimenter = Exp_OFDM_Declipping(args, plot_animation=False)

    # Load test dataset
    print(f"Loading test data from: {args.dset.path}")
    dataset = OFDMTestDatasetWithMetadata(
        args.dset,
        fs=44100,
        seg_len=args.audio_len,
        split='test'
    )

    print(f"Found {len(dataset)} test samples")

    # Test on a few samples
    num_samples = min(len(dataset), 5)  # Test on 5 samples
    results_list = []

    for idx in range(num_samples):
        sample = dataset[idx]

        signal = torch.from_numpy(sample['signal']).float()
        symbols = sample['symbols']

        ofdm_params = OFDMParams(
            fft_size=sample['fft_size'],
            cp_ratio=sample['cp_ratio'],
            guard_band_ratio=sample['guard_band_ratio'],
            modulation=sample['modulation']
        )

        print(f"\n--- Sample {idx+1}/{num_samples}: {sample['filename']} ---")
        print(f"  Modulation: {sample['modulation']}, FFT: {sample['fft_size']}")

        result = experimenter.conduct_experiment(
            signal, symbols, ofdm_params, name=f"test_{idx:03d}"
        )
        results_list.append(result)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    avg_evm_before = np.mean([r['evm_before_percent'] for r in results_list])
    avg_evm_after = np.mean([r['evm_after_percent'] for r in results_list])
    avg_improvement = np.mean([r['evm_improvement_percent'] for r in results_list])

    print(f"Average EVM before declipping: {avg_evm_before:.2f}%")
    print(f"Average EVM after declipping:  {avg_evm_after:.2f}%")
    print(f"Average EVM improvement:       {avg_improvement:.2f}%")
    print("=" * 60)

    return results_list


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)


@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)


if __name__ == "__main__":
    main()
