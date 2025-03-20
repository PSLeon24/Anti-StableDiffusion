# Anti-StableDiffusion
### Objective
In this work, adversarial perturbation 𝛿 via optimization algorithm is generated as follows:

![image](https://github.com/user-attachments/assets/99b86b64-478b-40cb-9e20-779de063af7f)

![image](https://github.com/user-attachments/assets/729e01d7-44fb-4c9a-b142-2b4fc44ccf0a)

- Example of (a) original image, (b) adversarial example, and (c) adversarial perturbations (5x amplified)

![image](https://github.com/user-attachments/assets/e2526533-8950-4db0-9e76-a3be05c20f5d)

### Experiments
- Evaluation Metrics: PSNR, SSIM, FID
  
- Quantitative Results
  
![image](https://github.com/user-attachments/assets/a6f0447a-4b4f-4e16-bdea-034650c03ca8)

- Qualitative Results
  
![image](https://github.com/user-attachments/assets/ce95850f-28e7-41ba-8fce-57823ac6f96c)

### Environment Setup

Pretrained checkpoints of different Stable Diffusion versions can be downloaded from the provided links in the table below:
|Version|Link|
|:--:|:--:|
|2.1|<a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a>|
|1.5|<a href="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5">stable-diffusion-v1-5</a>|
|1.4|<a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a>|

### Usage
```
Anti-StableDiffusionk/
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
- To be written.
