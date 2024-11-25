# 1. 필요한 라이브러리 임포트
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers.utils import load_image
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import lpips  # LPIPS: pip install lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import numpy as np
import random
import warnings
import time
import os
import csv

# FutureWarning 무시
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Hugging Face에서 Stable Diffusion 모델 로드
def load_model(model_name="stabilityai/stable-diffusion-2-1"):
    """Stable Diffusion img2img 파이프라인 모델 로드."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_auth_token=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

def load_txt2img_model(model_name="stabilityai/stable-diffusion-2-1"):
    """Stable Diffusion txt2img 파이프라인 모델 로드."""
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float16, use_auth_token=True)
    pipe.safety_checker = None
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe

# 4. Define Adversarial Attack against img2img
class Img2ImgAdversarialAttack:
    def __init__(self, model, adversarial_budget=16/255, alpha=1/255, steps=100, momentum_factor=0.9, non_mask_budget=4/255, non_mask_alpha=1/255):
        self.model = model
        self.adversarial_budget = adversarial_budget
        self.alpha = alpha
        self.steps = steps
        self.momentum_factor = momentum_factor
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)
        self.non_mask_budget = non_mask_budget
        self.non_mask_alpha = non_mask_alpha
        self.momentum = None

    def compute_denoising_loss(self, noisy_input, prompt):
        """Compute denoising loss between input with noise and denoised output."""
        denoised_output = self.model(prompt=prompt, image=noisy_input, strength=0.5, guidance_scale=5).images[0]
        denoised_tensor = transforms.ToTensor()(denoised_output).unsqueeze(0).to(device)
        denoising_loss = F.mse_loss(noisy_input, denoised_tensor)
        return denoising_loss

    def generate_adversarial(self, image, prompt):
        preprocess_input = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # 1. Segmentation using Segformer
        segformer_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        segformer_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device)
        segformer_inputs = segformer_processor(images=image, return_tensors="pt").to(device)
        segformer_outputs = segformer_model(**segformer_inputs)
        logits = segformer_outputs.logits
        upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        labels = upsampled_logits.argmax(dim=1)[0]

        # Define labels for specific regions
        desired_labels = [3, 4, 5,  6, 7, 10, 11, 12]
        combined_mask = torch.zeros_like(labels, dtype=torch.long).cpu()
        for label in desired_labels:
            combined_mask = combined_mask | (labels == label).long().cpu()

        # Convert mask to tensor
        combined_mask_tensor = combined_mask.unsqueeze(0).to(device, dtype=torch.float32)

        # 2. Prepare input tensor
        input_tensor = preprocess_input(image).unsqueeze(0).to(device).float()

        # Generate initial noise
        random_init = torch.rand_like(input_tensor) * self.adversarial_budget
        non_mask_random_init = torch.rand_like(input_tensor) * self.non_mask_budget
        random_init = (random_init * combined_mask_tensor) + (non_mask_random_init * (1 - combined_mask_tensor))
        perturbed_images = torch.clamp(input_tensor + random_init, 0, 1).requires_grad_(True)
        original_images = input_tensor.clone().detach()

        # Initialize momentum
        self.momentum = torch.zeros_like(input_tensor)

        # Initialize best loss and best adversarial image
        max_loss = -float('inf')
        strongest_adv_img = None

        # 3. Tokenize the prompt
        tokenizer = self.model.tokenizer
        input_ids = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        # Set to mixed precision for reduced memory usage
        weight_dtype = torch.float16

        for step in tqdm(range(self.steps), desc="Generating adversarial example"):
            if perturbed_images.grad is not None:
                perturbed_images.grad.zero_()
            if step % 10 != 0:
                continue

            perturbed_images_ = (perturbed_images + 1.0) / 2.0
            perturbed_images_ = torch.clamp(perturbed_images_, 0.0, 1.0)
            perturbed_images_ = perturbed_images_.to(dtype=weight_dtype)

            # 4. Forward diffusion and noise addition
            with torch.no_grad():
                latents = self.model.vae.encode(perturbed_images_).latent_dist.sample()
            noise = torch.randn_like(latents, dtype=weight_dtype)
            timesteps = torch.randint(0, self.model.scheduler.num_train_timesteps, (1,), device=latents.device).long()
            noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise residual
            encoder_hidden_states = self.model.text_encoder(input_ids).last_hidden_state
            model_pred = self.model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute losses
            lpips_loss = self.loss_fn(perturbed_images_, input_tensor.to(dtype=weight_dtype)).mean()
            denoising_loss = F.mse_loss(model_pred.float(), noise.float())

            # SSIM and brightness loss
            ssim_loss = 1 - ssim(perturbed_images_, input_tensor.to(dtype=weight_dtype), data_range=1.0)
            brightness_loss = torch.mean(torch.abs(perturbed_images_ - input_tensor.to(dtype=weight_dtype)))

            # original_image_tensor = input_tensor.clone().detach()
            # output_image = self.model(prompt=prompt, image=perturbed_images_, strength=0.5, guidance_scale=5).images[0]
            # output_tensor = preprocess_input(output_image).unsqueeze(0).to(device).float()
            # ssim_loss = 1 - ssim(output_tensor, original_image_tensor, data_range=1.0)
            # brightness_loss = torch.mean(torch.abs(output_tensor - original_image_tensor))

            combined_loss = 0.5 * denoising_loss + 0.3 * lpips_loss + 0.1 * ssim_loss + 0.1 * brightness_loss

            # Update the strongest adversarial image if the current loss is higher
            if combined_loss.item() > max_loss:
                max_loss = combined_loss.item()
                strongest_adv_img = perturbed_images.clone().detach()

            # Backpropagation
            combined_loss.backward()

            # Update perturbation
            with torch.no_grad():
                grad = perturbed_images.grad.clone()
                mask_grad = grad * combined_mask_tensor * self.alpha
                non_mask_grad = grad * (1 - combined_mask_tensor) * self.non_mask_alpha
                total_grad = mask_grad + non_mask_grad

                # Update momentum
                self.momentum = self.momentum_factor * self.momentum + total_grad / (torch.norm(total_grad) + 1e-8)
                perturbed_images = perturbed_images + self.alpha * self.momentum.sign()

                # Clamp perturbations
                mask_eta = torch.clamp(perturbed_images - original_images, -self.adversarial_budget, self.adversarial_budget)
                non_mask_eta = torch.clamp(perturbed_images - original_images, -self.non_mask_budget, self.non_mask_budget)
                perturbed_images = torch.clamp(
                    original_images + (mask_eta * combined_mask_tensor) + (non_mask_eta * (1 - combined_mask_tensor)), 
                    0, 1
                ).detach_().requires_grad_(True)

                # Clear GPU cache every 10 steps
                if step % 10 == 0:
                    torch.cuda.empty_cache()

            print(f"Step {step + 1}/{self.steps} | LPIPS Loss: {lpips_loss.item():.4f} | Denoising Loss: {denoising_loss.item():.4f} | SSIM Loss: {ssim_loss.item():.4f} | Brightness Loss: {brightness_loss:.4f} | Max Loss: {max_loss:.4f}")

        # Return the strongest adversarial image
        if strongest_adv_img is not None:
            adversarial_image = transforms.ToPILImage()(strongest_adv_img.squeeze().cpu())
        else:
            adversarial_image = transforms.ToPILImage()(perturbed_images.squeeze().cpu())
        return adversarial_image

# 5. Main
if __name__ == "__main__":
    # Set seed
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load the model
    model_name = "CompVis/stable-diffusion-v1-4"
    # "stabilityai/stable-diffusion-2-1"
    # "runwayml/stable-diffusion-v1-5"
    # "CompVis/stable-diffusion-v1-4"
    model = load_model(model_name)

    # Load the input image
    input_image = load_image("./Test/5956.jpg")
    prompt = "portrait of a person with a neutral expression"
    # "pink and blue hair, more larger eyes"
    # "portrait of a person with a neutral expression, purple hair"
    # "A high-quality portrait of a young person, highly detailed, realistic, smiling, with distinctive features."
    # "pink and blue hair"
    # "person with purple hair"
    # "A highly detailed portrait of a smiling person with exaggerated facial features, surreal lighting, and hyper-realistic details"

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, adversarial_budget=8/255, alpha=2/255, steps=100, momentum_factor=0.9, non_mask_budget=4/255, non_mask_alpha=1/255)
    start_time = time.time()
    adversarial_image = attack.generate_adversarial(input_image, prompt)
    end_time = time.time()

    print(f"Adversarial example generated in {end_time - start_time:.2f} seconds.")
    adversarial_image.show()
    adversarial_image.save("adversarial_image.png")

    # Generate outputs
    adversarial_output = model(prompt, image=adversarial_image, strength=0.5, guidance_scale=5).images[0] # model(prompt=prompt, image=adversarial_image, strength=0.4, guidance_scale=5.5).images[0]
    original_output= model(prompt, image=input_image, strength=0.5, guidance_scale=5).images[0]
    # original_output = model(prompt=prompt, image=input_image, strength=0.4, guidance_scale=5.5).images[0]

    # Save and show outputs
    adversarial_output.save("adversarial_output.png")
    adversarial_output.show()

    original_output.save("original_output.png")
    original_output.show()

    # LPIPS, SSIM, PSNR calculations
    input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
    adversarial_tensor = transforms.ToTensor()(adversarial_image).unsqueeze(0).to(device)
    original_output_tensor = transforms.ToTensor()(original_output).unsqueeze(0).to(device)
    adversarial_output_tensor = transforms.ToTensor()(adversarial_output).unsqueeze(0).to(device)

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_original = lpips_loss_fn(input_tensor, original_output_tensor).item()
    lpips_adversarial = lpips_loss_fn(adversarial_tensor, adversarial_output_tensor).item()

    ssim_original = ssim(original_output_tensor, input_tensor, data_range=1.0)
    psnr_original = psnr(original_output_tensor, input_tensor, data_range=1.0)

    ssim_adversarial = ssim(adversarial_output_tensor, adversarial_tensor, data_range=1.0)
    psnr_adversarial = psnr(adversarial_output_tensor, adversarial_tensor, data_range=1.0)

    # Print metrics
    print(f"Original LPIPS(↑): {lpips_original:.4f}")
    print(f"Adversarial LPIPS(↑): {lpips_adversarial:.4f}")
    print(f"Original SSIM(↓): {ssim_original:.4f}")
    print(f"Adversarial SSIM(↓): {ssim_adversarial:.4f}")
    print(f"Original PSNR(↓): {psnr_original:.4f}")
    print(f"Adversarial PSNR(↓): {psnr_adversarial:.4f}")


    # Calculate and visualize noise
    noise_tensor = adversarial_tensor - input_tensor
    noise_tensor = (noise_tensor - noise_tensor.min()) / (noise_tensor.max() - noise_tensor.min())  # Normalize to [0, 1]
    noise_image = transforms.ToPILImage()(noise_tensor.squeeze().cpu())
    noise_image.show()
    noise_image.save("noise_image.png")

    # Text-to-image example
    # txt2img_model = load_txt2img_model(model_name)
    # test_image = txt2img_model(prompt="person riding a bike", guidance_scale=5.5).images[0]
    # test_image.show()

    # 일단 주석 처리
    

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # CSV file to save metrics
    csv_file = "metrics.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Original_LPIPS", "Adversarial_LPIPS", "Original_SSIM", "Adversarial_SSIM", "Original_PSNR", "Adversarial_PSNR"])

    # Process all JPG files in the Test directory
    test_dir = "./Test"
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(".jpg"):
            file_path = os.path.join(test_dir, filename)

            base_filename = os.path.splitext(filename)[0]

            input_image = load_image(file_path)
            prompt = "portrait of a person with a neutral expression, purple hair"

            adversarial_image = attack.generate_adversarial(input_image, prompt)

            adversarial_output = model(prompt, image=adversarial_image, strength=0.5, guidance_scale=5).images[0]
            original_output = model(prompt, image=input_image, strength=0.5, guidance_scale=5).images[0]

            adversarial_output.save(os.path.join(output_dir, f"{base_filename}_adversarial_output.png"))
            original_output.save(os.path.join(output_dir, f"{base_filename}_original_output.png"))

            # Calculate metrics
            input_tensor = transforms.ToTensor()(input_image).unsqueeze(0).to(device)
            adversarial_tensor = transforms.ToTensor()(adversarial_image).unsqueeze(0).to(device)
            original_output_tensor = transforms.ToTensor()(original_output).unsqueeze(0).to(device)
            adversarial_output_tensor = transforms.ToTensor()(adversarial_output).unsqueeze(0).to(device)

            lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
            lpips_original = lpips_loss_fn(input_tensor, original_output_tensor).item()
            lpips_adversarial = lpips_loss_fn(adversarial_tensor, adversarial_output_tensor).item()

            ssim_original = ssim(original_output_tensor, input_tensor, data_range=1.0).item()
            psnr_original = psnr(original_output_tensor, input_tensor, data_range=1.0).item()

            ssim_adversarial = ssim(adversarial_output_tensor, adversarial_tensor, data_range=1.0).item()
            psnr_adversarial = psnr(adversarial_output_tensor, adversarial_tensor, data_range=1.0).item()

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([base_filename, lpips_original, lpips_adversarial, ssim_original, ssim_adversarial, psnr_original, psnr_adversarial])

            print(f"Processed {filename}: Metrics saved.")
            