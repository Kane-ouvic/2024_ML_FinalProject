# 2024 Machine Learning Final Project

This is a project for the ML course.

## Requirements

```
numpy>=1.21.0 
matplotlib>=3.4.0 
opencv-python>=4.5.0 
scikit-learn>=0.24.0 
pandas>=1.3.0 
torch>=1.10.0 
torchvision>=0.11.0 
torchaudio>=0.10.0
```

## Leaderboard Results

- **Rank**: 13/307

## Project Overview

This project focuses on enhancing fault detection in ultra-shallow seismic data using various machine learning techniques and data augmentation methods. The team experimented with multiple models and preprocessing approaches to improve segmentation accuracy.

### Key Approaches

1. **Data Augmentation**:

   - **Short-Time Fourier Transform (STFT)**: Removes high-frequency noise in seismic data but introduces additional noise due to discrete processing.
   - **Fast Fourier Transform (FFT)**: Filters high-frequency components in images to reduce noise, with reconstruction using Inverse FFT.
   - **Sobel Filter**: Highlights edges in seismic images for fault detection but is sensitive to noise and requires careful parameter tuning.
2. **Data Preprocessing**:

   - Sliced 3D seismic data into 2D images along the X, Y, and Z axes using a helper function (`slicing_xyz`).
   - Filtered out images without faults to reduce unnecessary data.
3. **Model Architectures**:

   - **Baseline Model**: Utilized a U-Net with contracting and expanding blocks.
   - **Simple U-Net**: Modified ResNet-18 as the encoder backbone.
   - **Attention U-Net**: Added spatial and channel-wise squeeze-and-excitation (scSE) modules to enhance feature focus.
   - **UNet++ and UNet++ Improved**:
     - Dense skip connections for feature reuse.
     - Incorporated Squeeze-and-Excitation (SE) blocks and replaced transposed convolutions with bilinear interpolation for improved segmentation.
4. **Ensemble Learning**:

   - Combined predictions from X, Y, and Z slices using intersection, union, or voting methods to leverage model strengths.

### Experimental Results


| Model           | Augmentation | Direction | Dice Score (Test) |
| --------------- | ------------ | --------- | ----------------- |
| Baseline        | None         | X         | 0.751948          |
| Simple U-Net    | None         | X         | 0.700252          |
| UNet++          | None         | X         | 0.770669          |
| Attention U-Net | None         | X         | 0.694207          |
| UNet++ Improved | None         | Voting    | **0.838948**      |

## Resources Used

- **Baseline**:
  - GPUs: 3 × RTX 4090
  - Training Time: 90 hours
- **UNet++ Improved**:
  - GPUs: 2 × V100
  - Training Time: 100 hours

## Key Findings

- Ensemble techniques significantly improved segmentation performance.
- The best results were achieved with the **UNet++ Improved** model using voting across X and Y directions.

## Discussion

- **Challenges**: Noise sensitivity in preprocessing methods, resource limitations during training.
- **Future Work**: Optimize preprocessing techniques and explore advanced ensemble methods.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Kane-ouvic/2024_ML_FinalProject.git
   ```
