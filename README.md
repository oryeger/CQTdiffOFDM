# Frequency-Domain Diffusion for OFDM Declipping

A diffusion-model-based approach for reconstructing clipped OFDM waveforms at the receiver, treating clipped samples as corrupted entries and learning a prior over clean OFDM signals.

## Overview

OFDM exhibits high peak-to-average power ratio (PAPR), motivating transmitter-side clipping to improve power amplifier efficiency. However, clipping introduces nonlinear distortion that degrades in-band constellation quality and increases Error Vector Magnitude (EVM). This project implements a diffusion model that operates on frequency-domain OFDM symbols to reconstruct clean waveforms from their clipped versions.

### Key Features

- **Frequency-domain diffusion denoiser** operating on equalized OFDM data symbols
- **EDM-style preconditioning** with continuous noise levels
- **Conditional generation** using distorted symbols as conditioning input
- **Warm-start sampling** from distorted symbols for faster convergence
- Support for **QPSK** and **16QAM** modulations

## Setup

This repository requires Python 3.8+ and PyTorch 1.10+. Other packages are listed in `requirements.txt`.

To install the requirements in your environment:

```bash
pip install -r requirements.txt
```

## Usage

### Training and Inference

Run the frequency-domain diffusion experiment with:

```bash
python experiment_freq_diffusion.py --clip_level 1.0 --train_steps 10000 --sampling_steps 150 --s_churn 2.5
```


## Architecture

The model consists of:

1. **Input Pipeline**: Time signal → Clipping → FFT Demodulation → Equalization → Distorted Symbols
2. **FreqDenoiser Model**: Conditional 1D residual network with sinusoidal sigma embedding
3. **EDM Preconditioning**: Scale-correct combination of network output with noisy input
4. **Sampling**: Stochastic Euler sampler with warm start from distorted symbols


## Authors

- Nicole Uzlaner
- Ory Eger

## Repository

GitHub: https://github.com/oryeger/CQTdiffOFDM

## References

- E. Moliner, J. Lehtinen, and V. Välimäki, "Solving Audio Inverse Problems with a Diffusion Model (CQT-Diff)", arXiv:2210.15228, 2022.
- Karras et al., "Elucidating the Design Space of Diffusion-Based Generative Models", arXiv:2206.00364, 2022.
- J. Ho, A. Jain, and P. Abbeel, "Denoising Diffusion Probabilistic Models", arXiv:2006.11239, 2020.
