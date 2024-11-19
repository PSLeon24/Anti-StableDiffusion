# Deepfake-Shield
### Objective
The goal of this project is to advance our research on disrupting deepfakes to a level where it can be presented at ICML 2025. This project aims to develop effective techniques that can interfere with the creation or dissemination of deepfake content, rendering it unusable or easily identifiable as manipulated.

![image](https://github.com/user-attachments/assets/4a83d5c5-6371-42e5-90c0-5774edbb7c9a)

### Abstract
Deepfake-Shield is a system designed to counter deepfake technology by introducing methods that actively disrupt the deepfake generation process or degrade the quality of deepfakes to make them less convincing. By leveraging adversarial techniques, noise injection, and data manipulation, this project provides a proactive approach to mitigating the risks posed by deepfake technology.

### Result
![image](https://github.com/user-attachments/assets/1177b9f3-b90e-4446-abad-efc6bae5575f)

### Environment Setup

Pretrained checkpoints of different Stable Diffusion versions can be downloaded from provided links in the table below:
|Version|Link|
|:--:|:--:|
|2.1|<a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a>|
|1.5|<a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5">stable-diffusion-v1-5</a>|
|1.4|<a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a>|

### Usage
```
stable-diffusion-attack/
├── DeepfakeShield.py            
├── Test/                  # Folder containing input test images
├── adversarial_image.png  # Generated adversarial example (output)
├── adversarial_output.png # Stable Diffusion output with adversarial input
└── README.md              
```

#### Clone this repository and Install dependencies
1. Clone the repository: ```git clone https://github.com/yourusername/Deepfake-Shield.git``` 
2. Install dependencies: ```pip install -r requirements.txt```

#### Dataset Preparation


### Citation

```
@inproceedings{Citation Key,
  title={Deepfake-Shield},
  author={Yeong-Min Ko},
  booktitle={to be added.},
  year={2025}
}
```

---

## 1. Requirements Analysis
### Objective
  - The objective is to design a system capable of generating adaptive adversarial examples to disrupt diverse versions of the Stable Diffusion img2img pipeline.
  - The system must:
    - Dynamically tune attack hyperparameters based on input image properties and loss behaviors.
    - Ensure visually imperceptible perturbations while maximizing disruption.
    - Support efficient computation for high-resolution image pipelines.
### Key Features
  - Adaptive tuning of attack parameters (epsilon, alpha, steps, momentum_factor) per image.
  - Utilization of perceptual (LPIPS), structural (SSIM), and brightness losses for optimization.
  - High compatibility with Hugging Face's StableDiffusionImg2ImgPipeline.
### System Requirements
  - Hardware: CUDA-enabled GPU with at least 8GB VRAM.
  - Software: Python 3.8+
  - Libraries: PyTorch, Hugging Face Diffusers, LPIPS, TorchMetrics, Pillow.

## 2. Design
- The system is designed to disrupt the Stable Diffusion img2img pipeline using an adaptive approach. The key modules and their interactions are shown in the following diagram:
![mysystem](https://github.com/user-attachments/assets/f6badce7-0d82-4db8-9dd3-3b01919321d8)

### Modules
- Image Preprocessing:
  - Resizes and normalizes input images for compatibility with Stable Diffusion pipelines.
- Stable Diffusion Pipeline:
  - A pre-trained img2img model from Hugging Face.
- Loss Functions:
  - Combines LPIPS, SSIM, and brightness losses to optimize perturbations.
- Adaptive Hyperparameter Module:
  - Dynamically tunes attack parameters based on image complexity and loss gradients.
- Perturbation Generator:
  - Iteratively updates input tensors to apply adversarial perturbations.
### Workflow
- Input Image Processing:
  - Preprocess input to match Stable Diffusion requirements (512x512 resolution, RGB).
- Loss-Based Optimization:
  - Calculate perceptual and structural losses between the input and output images.
- Dynamic Parameter Adjustment:
  - Adjust hyperparameters (epsilon, alpha, steps, momentum_factor) during the attack.
- Adversarial Image Generation:
  - Generate adversarial examples iteratively and save results.


## 3. Implementation

## 4. Testing
### Metrics:
 - LPIPS Distance: Evaluates perceptual similarity between original and adversarial outputs.
 - SSIM Score: Measures structural similarity between original and adversarial outputs.

## 5. Maintenance

### References
[1] Hugging Face Diffusers Documentation: https://huggingface.co/docs/diffusers
[2] LPIPS (Learned Perceptual Image Patch Similarity): https://github.com/richzhang/PerceptualSimilarity
[3] PyTorch Documentation: https://pytorch.org
