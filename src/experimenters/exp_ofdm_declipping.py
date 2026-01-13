"""
OFDM Declipping Experimenter.

Conducts declipping experiments on OFDM signals with EVM-based evaluation.
"""

import os
from datetime import date
import torch
import numpy as np

from src.sampler_ofdm import SamplerOFDMDeclipping
from src.ofdm.ofdm_generator import OFDMParams, demodulate_ofdm
from src.utils.evm import compute_evm, compute_evm_db
from src.models.unet_ofdm import Unet_OFDM
from src.utils.setup import load_ema_weights
from src.sde import VE_Sde_Elucidating


class Exp_OFDM_Declipping:
    """
    Experimenter for OFDM signal declipping with EVM evaluation.
    """

    def __init__(self, args, plot_animation=False):
        """
        Args:
            args: Hydra configuration
            plot_animation: Whether to return intermediate results for animation
        """
        self.args = args
        self.__plot_animation = plot_animation

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        if self.args.architecture == "unet_ofdm":
            self.model = Unet_OFDM(self.args, self.device).to(self.device)
        elif self.args.architecture == "unet_1d":
            from src.models.unet_1d import Unet_1d
            self.model = Unet_1d(self.args, self.device).to(self.device)
        elif self.args.architecture == "unet_CQT":
            from src.models.unet_cqt import Unet_CQT
            self.model = Unet_CQT(self.args, self.device).to(self.device)
        else:
            raise NotImplementedError(f"Architecture {self.args.architecture} not supported")

        # Load weights
        checkpoint_path = os.path.join(args.model_dir, args.inference.checkpoint)
        self.model = load_ema_weights(self.model, checkpoint_path)

        # Diffusion parameters
        if args.sde_type == 'VE_elucidating':
            self.diff_parameters = VE_Sde_Elucidating(
                self.args.diffusion_parameters,
                self.args.diffusion_parameters.sigma_data
            )
        else:
            raise NotImplementedError

        torch.backends.cudnn.benchmark = True

        # Get guidance parameters
        xi = getattr(args.inference, 'xi', 1.0)
        xi_evm = getattr(args.inference, 'xi_evm', 1.0)

        # Create sampler with combined guidance
        self.sampler = SamplerOFDMDeclipping(
            self.model,
            self.diff_parameters,
            self.args,
            xi=xi,
            xi_evm=xi_evm,
            order=2,
            data_consistency=args.inference.data_consistency,
            rid=self.__plot_animation
        )

        # Setup output paths
        today = date.today()
        self.path_sampling = os.path.join(
            args.model_dir,
            "ofdm_declipping_" + today.strftime("%d_%m_%Y")
        )
        if not os.path.exists(self.path_sampling):
            os.makedirs(self.path_sampling)

        # Clipping specific paths
        sdr = getattr(self.args.inference.declipping, 'SDR', 3)
        clip_ratio = getattr(self.args.inference.declipping, 'clip_ratio', None)

        if clip_ratio:
            n = f"clipped_ratio_{clip_ratio}"
        else:
            n = f"clipped_SDR_{sdr}"

        self.path_degraded = os.path.join(self.path_sampling, n)
        if not os.path.exists(self.path_degraded):
            os.makedirs(self.path_degraded)

        self.path_original = os.path.join(self.path_sampling, "original")
        if not os.path.exists(self.path_original):
            os.makedirs(self.path_original)

        self.path_reconstructed = os.path.join(self.path_sampling, "declipped")
        if not os.path.exists(self.path_reconstructed):
            os.makedirs(self.path_reconstructed)

    def get_clip_value_from_ratio(self, signal, clip_ratio):
        """Get clip value from ratio of peak amplitude."""
        peak = torch.max(torch.abs(signal))
        return peak * clip_ratio

    def get_clip_value_from_sdr(self, signal, target_sdr):
        """
        Find clip value that achieves target SDR.

        SDR = 10 * log10(signal_power / distortion_power)
        """
        from scipy.optimize import brentq

        signal_np = signal.cpu().numpy()

        def compute_sdr(clip_val):
            clipped = np.clip(signal_np, -clip_val, clip_val)
            distortion = signal_np - clipped
            signal_power = np.mean(signal_np ** 2)
            distortion_power = np.mean(distortion ** 2) + 1e-10
            return 10 * np.log10(signal_power / distortion_power) - target_sdr

        # Find clip value using root finding
        peak = np.max(np.abs(signal_np))
        try:
            clip_value = brentq(compute_sdr, 0.01 * peak, 0.99 * peak)
        except ValueError:
            # Fall back to ratio if SDR not achievable
            clip_value = 0.5 * peak

        return clip_value

    def conduct_experiment(
        self,
        signal: torch.Tensor,
        symbols: np.ndarray,
        ofdm_params: OFDMParams,
        name: str
    ):
        """
        Conduct declipping experiment on OFDM signal.

        Args:
            signal: Time-domain OFDM signal, shape (T,) or (B, T)
            symbols: Original QAM symbols for EVM reference
            ofdm_params: OFDM parameters used for generation
            name: Experiment name for saving results

        Returns:
            Dictionary with results including EVM before/after
        """
        # Ensure proper shape
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        signal = signal.to(self.device)

        # Get clipping parameters
        clip_ratio = getattr(self.args.inference.declipping, 'clip_ratio', None)
        target_sdr = getattr(self.args.inference.declipping, 'SDR', 3)

        if clip_ratio:
            clip_value = self.get_clip_value_from_ratio(signal, clip_ratio)
        else:
            clip_value = self.get_clip_value_from_sdr(signal[0], target_sdr)

        # Apply clipping
        y_clipped = torch.clip(signal, min=-clip_value, max=clip_value)

        # Compute EVM before declipping
        num_ofdm_symbols = symbols.shape[0]
        clipped_symbols = demodulate_ofdm(y_clipped[0].cpu().numpy(), ofdm_params, num_ofdm_symbols)
        evm_before = compute_evm(symbols, clipped_symbols)
        evm_db_before = compute_evm_db(symbols, clipped_symbols)

        # Run declipping
        if self.__plot_animation:
            x_hat, data_denoised, t = self.sampler.predict_ofdm_declipping(
                y_clipped, clip_value, symbols, ofdm_params
            )
        else:
            x_hat = self.sampler.predict_ofdm_declipping(
                y_clipped, clip_value, symbols, ofdm_params
            )

        # Compute EVM after declipping
        declipped_symbols = demodulate_ofdm(x_hat[0].cpu().numpy(), ofdm_params, num_ofdm_symbols)
        evm_after = compute_evm(symbols, declipped_symbols)
        evm_db_after = compute_evm_db(symbols, declipped_symbols)

        # Save results
        results = {
            'name': name,
            'clip_value': float(clip_value),
            'evm_before_percent': evm_before,
            'evm_after_percent': evm_after,
            'evm_before_db': evm_db_before,
            'evm_after_db': evm_db_after,
            'evm_improvement_percent': evm_before - evm_after,
            'evm_improvement_db': evm_db_before - evm_db_after,
            'modulation': ofdm_params.modulation,
            'fft_size': ofdm_params.fft_size,
            'cp_ratio': ofdm_params.cp_ratio,
            'guard_band_ratio': ofdm_params.guard_band_ratio,
        }

        # Save signals
        np.save(
            os.path.join(self.path_original, f"{name}_original.npy"),
            signal[0].cpu().numpy()
        )
        np.save(
            os.path.join(self.path_degraded, f"{name}_clipped.npy"),
            y_clipped[0].cpu().numpy()
        )
        np.save(
            os.path.join(self.path_reconstructed, f"{name}_declipped.npy"),
            x_hat[0].cpu().numpy()
        )

        print(f"\n{name}:")
        print(f"  EVM before: {evm_before:.2f}% ({evm_db_before:.2f} dB)")
        print(f"  EVM after:  {evm_after:.2f}% ({evm_db_after:.2f} dB)")
        print(f"  Improvement: {evm_before - evm_after:.2f}% ({evm_db_before - evm_db_after:.2f} dB)")

        if self.__plot_animation:
            results['data_denoised'] = data_denoised
            results['t'] = t

        return results


def run_ofdm_declipping_experiment(args):
    """
    Run OFDM declipping experiment from command line.
    """
    from src.ofdm.ofdm_dataset_loader import OFDMTestDatasetWithMetadata

    # Create experimenter
    experimenter = Exp_OFDM_Declipping(args, plot_animation=False)

    # Load test dataset
    dataset = OFDMTestDatasetWithMetadata(
        args.dset,
        fs=args.sample_rate,
        seg_len=args.audio_len,
        split='test'
    )

    results_list = []

    for idx in range(min(len(dataset), 10)):  # Test on first 10 samples
        sample = dataset[idx]

        signal = torch.from_numpy(sample['signal']).float()
        symbols = sample['symbols']

        ofdm_params = OFDMParams(
            fft_size=sample['fft_size'],
            cp_ratio=sample['cp_ratio'],
            guard_band_ratio=sample['guard_band_ratio'],
            modulation=sample['modulation']
        )

        result = experimenter.conduct_experiment(
            signal, symbols, ofdm_params, name=sample['filename']
        )
        results_list.append(result)

    # Print summary
    print("\n" + "=" * 50)
    print("Summary:")
    avg_evm_before = np.mean([r['evm_before_percent'] for r in results_list])
    avg_evm_after = np.mean([r['evm_after_percent'] for r in results_list])
    print(f"Average EVM before: {avg_evm_before:.2f}%")
    print(f"Average EVM after:  {avg_evm_after:.2f}%")
    print(f"Average improvement: {avg_evm_before - avg_evm_after:.2f}%")

    return results_list
