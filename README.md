# Anti-StableDiffusion
### Objective
In this work, adversarial perturbation 𝛿 via optimization algorithm is generated as follows:

![image](https://github.com/user-attachments/assets/99b86b64-478b-40cb-9e20-779de063af7f)

![image](https://github.com/user-attachments/assets/729e01d7-44fb-4c9a-b142-2b4fc44ccf0a)

- Example of (a) original image, (b) adversarial example, and (c) adversarial perturbations (5x amplified)

![image](https://github.com/user-attachments/assets/e2526533-8950-4db0-9e76-a3be05c20f5d)

### Result
- Quantitative Results
  
![image](https://github.com/user-attachments/assets/a6f0447a-4b4f-4e16-bdea-034650c03ca8)

- Qualitative Results
  
![image](https://github.com/user-attachments/assets/ce95850f-28e7-41ba-8fce-57823ac6f96c)

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
- The system is designed to disrupt the Stable Diffusion img2img pipeline using an adaptive approach.

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

## 3. Metrics:
 - PSNR, SSIM, FID

