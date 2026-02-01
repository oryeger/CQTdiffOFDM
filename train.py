"""

Main script for training
"""
import os
import hydra
import platform

import torch

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np


def get_device():
    """Get the best available device: CUDA > MPS (Mac) > CPU"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU (training will be slow)")
        print("Tip: Upgrade PyTorch to 1.12+ for MPS support on Mac: pip install --upgrade torch")
    return device


def worker_init_fn(worker_id):
    """Worker init function for DataLoader (must be at module level for Windows)."""
    st = np.random.get_state()[2]
    np.random.seed(st + worker_id)


def run(args):
    """Loads all the modules and starts the training

    Args:
      args:
        Hydra dictionary

    """

    #some preparation of the hydra args
    args = OmegaConf.structured(OmegaConf.to_yaml(args))

    #choose best available device (CUDA > MPS > CPU)
    device = get_device()

    dirname = os.path.dirname(__file__)

    #define the path where weights will be loaded and saved
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    # Enable cudnn benchmark only for CUDA
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    
    # Adjust num_workers for Mac (multiprocessing issues)
    if platform.system() == "Darwin":  # macOS
        if args.num_workers > 0:
            print(f"Mac detected: reducing num_workers from {args.num_workers} to 0 for stability")
            args.num_workers = 0

    print("Training on: ",args.dset.name)

    # Check if EVM-aware training is enabled
    lambda_evm = getattr(args, 'lambda_evm', 0.0)
    use_evm_training = (args.dset.name == "ofdm" and lambda_evm > 0)
    
    if use_evm_training:
        print(f"EVM-aware training ENABLED (lambda_evm={lambda_evm})")
    
    #prepare the dataset loader
    if args.dset.name == "ofdm":
        if use_evm_training:
            # Use OFDM dataset with symbols for EVM loss
            from src.ofdm.ofdm_dataset_loader import OFDMTrainDatasetWithSymbols
            dataset_train = OFDMTrainDatasetWithSymbols(args.dset, args.sample_rate * args.resample_factor, args.audio_len * args.resample_factor)
        else:
            # Use standard OFDM dataset (no symbols)
            from src.ofdm.ofdm_dataset_loader import OFDMTrainDataset
            dataset_train = OFDMTrainDataset(args.dset, args.sample_rate * args.resample_factor, args.audio_len * args.resample_factor)
    else:
        # Use standard audio dataset loader
        import src.dataset_loader as dataset
        dataset_train = dataset.TrainDataset(args.dset, args.sample_rate * args.resample_factor, args.audio_len * args.resample_factor)

    train_loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, worker_init_fn=worker_init_fn)
    train_set = iter(train_loader)

    #prepare the model architecture

    if args.architecture == "unet_CQT":
        from src.models.unet_cqt import Unet_CQT
        model = Unet_CQT(args, device).to(device)
    elif args.architecture == "unet_STFT":
        from src.models.unet_stft import Unet_STFT
        model = Unet_STFT(args, device).to(device)
    elif args.architecture == "unet_1d":
        from src.models.unet_1d import Unet_1d
        model = Unet_1d(args, device).to(device)
    elif args.architecture == "unet_ofdm":
        from src.models.unet_ofdm import Unet_OFDM
        model = Unet_OFDM(args, device).to(device)
    else:
        raise NotImplementedError(f"Architecture {args.architecture} not supported")

    #prepare the learner (optimizer is inside)

    if use_evm_training:
        # Use OFDM-aware learner with EVM loss
        from src.learner_ofdm import LearnerOFDM
        learner = LearnerOFDM(
            args.model_dir, model, train_set, args, log=args.log
        )
    else:
        # Use standard learner
        from src.learner import Learner
        learner = Learner(
            args.model_dir, model, train_set, args, log=args.log
        )

    #start the training
    learner.train()


def _main(args):
    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    run(args)

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()
