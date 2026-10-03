# Diffusion MNIST

A diffusion-based video generation project for **Moving MNIST**, using a two-stage latent video generation pipeline with a **Variational Autoencoder (VAE)** and a **Diffusion Transformer for Video (DITVideo)**.

The implementation follows the architecture presented in **Latte: Latent Diffusion Transformer for Video Generation**.

## Project Overview

The project uses a two-stage approach:

1. **Variational Autoencoder (VAE)** — compresses video frames into a compact latent representation and reconstructs them.
2. **Diffusion Transformer for Video (DITVideo)** — learns the diffusion process in latent space and generates new video representations.

The model is trained and evaluated on the **Moving MNIST Easy** dataset, consisting of videos of handwritten digits moving across a sequence of frames.

### Pipeline

```text
Moving MNIST Videos
        ↓
Frame Extraction
        ↓
VAE Training
        ↓
Latent Generation
        ↓
DITVideo Training
        ↓
Video Sampling
        ↓
Generated Videos
```

## Dataset

This project uses the **Captioned Moving MNIST Dataset — Easy Version**, containing videos of single MNIST digits moving across frames.

Download the dataset from Kaggle and place the video files in:

```text
data/mmnist-easy/
```

The extracted frames are stored under:

```text
mmnist/mmnist-easy-images/
```

## Project Structure

```text
├── config/
│   └── mnist.yaml              # Configuration for the Moving MNIST pipeline
│
├── dataset/
│   ├── image_dataset.py         # Dataset for VAE frame training
│   └── video_dataset.py        # Dataset for diffusion video training
│
├── model/
│   ├── attention.py             # Attention components
│   ├── blocks.py                # VAE and transformer building blocks
│   ├── discriminator.py         # VAE adversarial training
│   ├── lpips.py                # Perceptual loss
│   ├── patch_embed.py           # Video patch embedding
│   ├── transformer.py           # DITVideo implementation
│   ├── transformer_layer.py     # Transformer layers
│   ├── vae.py                   # VAE implementation
│   └── weights/                 # Model weights
│
├── scheduler/
│   └── linear_scheduler.py      # Diffusion noise scheduler
│
├── tools/
│   ├── sample_vae_ditv.py       # Generate video samples
│   ├── save_latents.py          # Pre-compute latent representations
│   ├── train_vae.py             # Train the VAE
│   └── train_vae_ditv.py        # Train the diffusion model
│
├── extract_frames.py            # Extract frames from videos
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

The project uses PyTorch and is intended for GPU-based training.

## Usage

### 1. Extract Frames

Extract individual frames from the Moving MNIST videos:

```bash
python extract_frames.py --config config/mnist.yaml
```

### 2. Train the VAE

Train the Variational Autoencoder:

```bash
python -m tools.train_vae --config config/mnist.yaml
```

The VAE learns a latent representation that is used by the subsequent diffusion stage.

### 3. Generate Latents

Pre-compute latent representations for diffusion model training:

```bash
python -m tools.save_latents --config config/mnist.yaml
```

### 4. Train the Diffusion Model

Train the Diffusion Transformer for Video:

```bash
python -m tools.train_vae_ditv --config config/mnist.yaml
```

### 5. Generate Videos

Generate video samples using the trained VAE and diffusion model:

```bash
python -m tools.sample_vae_ditv --config config/mnist.yaml
```

Generated samples are reconstructed from the generated latent representations using the VAE decoder.

## Configuration

All major dataset, model, diffusion, and training parameters are controlled through:

```text
config/mnist.yaml
```

The configuration includes:

* `dataset_params` — dataset loading and preprocessing
* `diffusion_params` — diffusion process parameters
* `ditv_params` — DITVideo architecture parameters
* `autoencoder_params` — VAE architecture parameters
* `train_params` — training settings and checkpoint paths

## Model Architecture

### Variational Autoencoder

The VAE learns a compact latent representation of video frames.

Key components include:

* Hierarchical encoder with downsampling blocks
* Latent mean and log-variance prediction
* Decoder for frame reconstruction
* Reconstruction loss
* KL-divergence loss
* Perceptual loss
* Adversarial loss

### Diffusion Transformer for Video

DITVideo performs diffusion in the learned latent space.

Key components include:

* Video patch embeddings
* Spatial and temporal attention
* Transformer blocks
* Diffusion timestep embeddings
* Noise prediction for iterative denoising

The resulting latent representation is decoded by the VAE to produce video frames.

## Technologies

* Python
* PyTorch
* Variational Autoencoder (VAE)
* Diffusion Models
* Diffusion Transformer
* Self-Attention
* Moving MNIST
* CUDA

## Citation

This implementation follows the approach presented in **Latte: Latent Diffusion Transformer for Video Generation**.

```bibtex
@misc{ma2024lattelatentdiffusiontransformer,
      title={Latte: Latent Diffusion Transformer for Video Generation},
      author={Xin Ma and Yaohui Wang and Gengyun Jia and Xinyuan Chen and Ziwei Liu and Yuan-Fang Li and Cunjian Chen and Yu Qiao},
      year={2024},
      eprint={2401.03048},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2401.03048}
}
```

### Dataset Reference

The project uses the Moving MNIST dataset introduced in:

```bibtex
@article{srivastava2015unsupervised,
  title={Unsupervised Learning of Video Representations using LSTMs},
  author={Srivastava, Nitish and Mansimov, Alexey and Salakhutdinov, Ruslan},
  journal={Proceedings of the 32nd International Conference on Machine Learning},
  year={2015}
}
```

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
