# Neural Image Reconstruction and Processing Pipeline

This repository provides tools and scripts for processing the Natural Scenes Dataset (NSD), extracting features, training regression models, and reconstructing images using state-of-the-art deep learning models.

---

## Prerequisites

Ensure the following tools and libraries are installed on your system:

- Python 3.8+
- pip
- AWS CLI
- NVIDIA GPU with CUDA support
- Basic Linux utilities (`wget`, `curl`, `unzip`, etc.)

---

## Steps to Run

### 1. Install Required Libraries
```bash
pip install -r requirements.txt
```

### 2. Download NSD Data

1. Install AWS CLI (if not already installed) and log in. This is required to access the NSD datasets stored in a public bucket.
2. Navigate to the `data/` directory:
    ```bash
    cd data/
    ```
3. Download the NSD data:
    ```bash
    python3 download_nsdata.py
    ```

### 3. Download COCO Dataset with Captions

1. Navigate to the `annots/` directory:
    ```bash
    cd annots/
    ```
2. Download the dataset:
    ```bash
    wget https://huggingface.co/datasets/pscotti/naturalscenesdataset/resolve/main/COCO_73k_annots_curated.npy
    ```

### 4. Prepare NSD Data Using Nibabel

1. Navigate back to the parent directory:
    ```bash
    cd ..
    ```
2. Prepare the data for all subjects (1, 2, 5, 7):
    ```bash
    python3 prepare_nsddata.py -sub x
    ```

### 5. Download VDVAE Model

1. Navigate to the `vdvae/model` directory:
    ```bash
    cd vdvae/model
    ```
2. Download the pre-trained VDVAE model files:
    ```bash
    wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-log.jsonl
    wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model.th
    wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th
    wget https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-opt.th
    ```

### 6. Train FMRI-Z Regressor Using VDVAE

1. Navigate back to the main directory:
    ```bash
    cd ../..
    ```
2. Extract features:
    ```bash
    python3 scripts/vdvae_extract_features.py -sub x
    ```
3. Train the regressor:
    ```bash
    python3 scripts/vdvae_regression.py -sub x
    ```

### 7. Reconstruct Base Image Using Trained Regressor and VDVAE Decoder

```bash
python3 scripts/vdvae_reconstruct_images.py -sub x
```

### 8. Download Versatile Diffusion Model

1. Navigate to the `versatile_diffusion/pretrained/` directory:
    ```bash
    cd versatile_diffusion/pretrained/
    ```
2. Download the pre-trained Versatile Diffusion model files:
    ```bash
    wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/vd-four-flow-v1-0-fp16-deprecated.pth
    wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/kl-f8.pth
    wget https://huggingface.co/shi-labs/versatile-diffusion/resolve/main/pretrained_pth/optimus-vae.pth
    ```

### 9. Train Vision and Text Regressors

1. Navigate back to the main directory:
    ```bash
    cd ../..
    ```
2. Extract features:
    ```bash
    python3 scripts/cliptext_extract_features.py -sub x
    python3 scripts/clipvision_extract_features.py -sub x
    ```
3. Train the regressors:
    ```bash
    python3 scripts/cliptext_regression.py -sub x
    python3 scripts/clipvision_regression.py -sub x
    ```

### 10. Reconstruct Final Images Using Diffusion

1. Reconstruct images:
    ```bash
    python3 scripts/versatilediffusion_reconstruct_images.py -sub x
    ```
2. Save test images:
    ```bash
    python3 scripts/save_test_images.py
    ```

### 11. Compare Images

Compare the reconstructed images in the following folders:

- `/data/nsddata_stimuli/test_data`
- `/results/vdvae/sub0x`
- `/results/versatile_diffusionsub0x`
