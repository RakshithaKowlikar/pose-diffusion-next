# PoseDiffusion-Next

Lightweight experiments on 3D human-pose forecasting using baseline UNet and Transformer architectures on the AMASS dataset.  
Work in progress — the next stage will explore diffusion-based and physics-aware models for temporally consistent motion prediction.

---

## Overview

This repository explores short-horizon 3D human-pose forecasting through baseline architectures and an early diffusion framework.  
The primary goal is to build an efficient, interpretable, and lightweight motion-prediction model optimized for limited GPU memory.

Implemented components:

- **UNet** – temporal convolutional baseline for smooth motion prediction  
- **Transformer** – sequence-to-sequence baseline with positional encoding  
- **Diffusion framework** – forward/reverse noise scheduling, temporal denoisers, and sampling loop  
- **TemporalUNet** and **TransformerDenoiser** – denoising modules for diffusion-based forecasting  

Dataset: [AMASS](https://amass.is.tue.mpg.de/) (CMU, KIT, Transitions subsets), represented in **SMPL-H joint coordinates**.  
Each sequence contains 3-D joint trajectories extracted from motion-capture recordings.

---

## Technical Details

- **Frameworks:** PyTorch, NumPy, Matplotlib  
- **Input representation:** SMPL-H joints (22 × 3)  
- **Forecasting setup:** Given past frames, predict future joint trajectories  
- **Training objective:** Mean-squared error between predicted and ground-truth joint positions  
- **Metrics:** Mean Per-Joint Position Error (MPJPE)

---

## Next Steps

- Integrate a full diffusion-based denoising model that includes flow matching 
- Add physics constraints (bone-length consistency, velocity smoothness)  
- Evaluate long-horizon stability and multi-modal future sampling  
- Develop lightweight training strategies for low-VRAM hardware (RTX 4050)

---

## Acknowledgements

- **Dataset:** AMASS  
- **Body model:** SMPL-H by MPI-IS  
- **Libraries:** PyTorch, PyTorch3D, NumPy, Matplotlib  

---
