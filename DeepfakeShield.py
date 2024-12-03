# 1. 필요한 라이브러리 임포트
import torch
import torch.nn.functional as F
from torch import nn
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, DDIMScheduler
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from diffusers.utils import load_image
import torchvision
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
    pipe.enable_attention_slicing()  # Attention Slicing 활성화
    pipe.enable_xformers_memory_efficient_attention()
    # pipe.enable_xformers_memory_efficient_attention()  # 메모리 효율적 Attention
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
    def __init__(self, model, vae, tokenizer, noise_scheduler, adversarial_budget=16/255, alpha=1/255, steps=100, momentum_factor=0.9, non_mask_budget=4/255, non_mask_alpha=0.5/255, random_start=True):
        self.model = model
        self.vae = vae
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        self.adversarial_budget = adversarial_budget
        self.alpha = alpha
        self.steps = steps
        self.momentum_factor = momentum_factor
        self.non_mask_budget = non_mask_budget
        self.non_mask_alpha = non_mask_alpha
        self.random_start = random_start
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)  # LPIPS 손실
        self.momentum = None
        self.model.enable_attention_slicing()
        self.model.enable_xformers_memory_efficient_attention()

    def compute_loss(self, perturbed_images, input_tensor, model_pred, target_noise, mask_tensor, latent_original):
        """결합된 손실 함수: LPIPS + TV 손실 + Diffusion 손실 + Gaussian Augmentation + Latent Feature Distortion"""
        # LPIPS Loss (Perceptual 손실)
        lpips_loss = self.loss_fn(perturbed_images, input_tensor).mean()

        # TV Loss (Total Variation 손실)
        tv_loss = torch.mean(torch.abs(perturbed_images[:, :, :-1] - perturbed_images[:, :, 1:])) + \
                torch.mean(torch.abs(perturbed_images[:, :-1, :] - perturbed_images[:, 1:, :]))

        # Diffusion Loss
        diffusion_loss = F.mse_loss(model_pred, target_noise)

        if mask_tensor is not None:
            masked_lpips_loss = torch.sum(lpips_loss * mask_tensor) / torch.sum(mask_tensor)
            masked_diffusion_loss = torch.sum(diffusion_loss * mask_tensor) / torch.sum(mask_tensor)
        else:
            masked_lpips_loss = lpips_loss
            masked_diffusion_loss = diffusion_loss

        # Gaussian Augmentation Loss
        gaussian_perturbed = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))(perturbed_images)
        gaussian_loss = F.mse_loss(gaussian_perturbed, input_tensor)

        # Latent Feature Distortion Loss
        with torch.no_grad():
            latent_perturbed = self.vae.encode(perturbed_images.to(dtype=torch.float16)).latent_dist.mean.to(dtype=torch.float32)
        latent_loss = F.mse_loss(latent_original, latent_perturbed)

        # 결합 손실
        combined_loss = (
            0.5 * (masked_lpips_loss + masked_diffusion_loss) +
            0.2 * (((1 - mask_tensor).mean() * lpips_loss) + ((1 - mask_tensor).mean() * diffusion_loss)) +
            0.1 * tv_loss +
            0.1 * gaussian_loss +
            0.1 * latent_loss
        )
        return combined_loss

    def generate_adversarial(self, image, prompt, opposite_prompt="a distorted object"):
        preprocess_input = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # 1. 입력 텐서 준비
        input_tensor = preprocess_input(image).unsqueeze(0).to(device).float()

        # 2. 랜덤 초기화 (랜덤 시작 활성화 시)
        if self.random_start:
            random_noise = torch.rand_like(input_tensor) * self.adversarial_budget
            input_tensor = torch.clamp(input_tensor + random_noise, 0, 1)

        perturbed_images = input_tensor.clone().detach().requires_grad_(True)

        # 3. 텍스트 프롬프트 토큰화
        input_ids = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        # 4. 반대 프롬프트 토큰화
        opposite_input_ids = self.tokenizer(
            opposite_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)

        # 5. 텍스트 인코더 출력 생성
        encoder_hidden_states = self.model.text_encoder(input_ids).last_hidden_state
        opposite_hidden_states = self.model.text_encoder(opposite_input_ids).last_hidden_state

        # 텍스트 인코더에 교란 추가: 정반대 방향으로 임베딩을 왜곡
        attack_strength = 0.3  # 교란 강도
        perturbed_hidden_states = encoder_hidden_states + attack_strength * (opposite_hidden_states - encoder_hidden_states)

        # 6. 마스킹 텐서 생성 (Segformer 사용)
        segformer_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        segformer_model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing").to(device)
        segformer_inputs = segformer_processor(images=image, return_tensors="pt").to(device)
        segformer_outputs = segformer_model(**segformer_inputs)
        logits = segformer_outputs.logits
        upsampled_logits = nn.functional.interpolate(logits, size=image.size[::-1], mode='bilinear', align_corners=False)
        labels = upsampled_logits.argmax(dim=1)[0]

        desired_labels = [3, 4, 5, 6, 7, 10, 11, 12, 13]  # 얼굴 영역 (예: 눈, 입, 헤어 등)
        combined_mask = torch.zeros_like(labels, dtype=torch.long).cpu()
        for label in desired_labels:
            combined_mask = combined_mask | (labels == label).long().cpu()

        # 마스크를 텐서로 변환
        mask_tensor = combined_mask.unsqueeze(0).to(device, dtype=torch.float32)

        max_loss = -float('inf')
        best_adv_img = None

        # 모멘텀 초기화
        momentum = torch.zeros_like(input_tensor).to(device)

        for step in tqdm(range(self.steps), desc="Generating adversarial example"):
            torch.cuda.empty_cache() 
            if perturbed_images.grad is not None:
                perturbed_images.grad.zero_()
            if step % 10 != 0:
                # torch.cuda.empty_cache()  # 주기적으로 캐시 해제
                continue

            perturbed_images_ = torch.clamp(perturbed_images, 0.0, 1.0)

            # VAE의 데이터 타입을 float16으로 변환
            perturbed_images_ = perturbed_images_.to(dtype=torch.float16)  # 데이터 타입 변환

            # 5. VAE 인코딩 및 노이즈 추가
            with torch.no_grad():
                latent_original = self.vae.encode(input_tensor.to(dtype=torch.float16)).latent_dist.mean.to(dtype=torch.float32)
            with torch.no_grad():
                latents = self.vae.encode(perturbed_images_).latent_dist.sample()
                latents = latents * self.vae.config.scaling_factor

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), device=latents.device).long()
            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

            # 6. U-Net 예측
            model_pred = self.model.unet(noisy_latents, timesteps, perturbed_hidden_states).sample

            # 7. 손실 계산
            perturbed_images_ = perturbed_images_.to(dtype=torch.float32)
            loss = self.compute_loss(perturbed_images_, input_tensor, model_pred, noise, mask_tensor, latent_original)


            # 최적의 적대적 예제 저장
            if loss.item() > max_loss:
                max_loss = loss.item()
                best_adv_img = perturbed_images.clone().detach()

            # 8. 역전파 및 업데이트
            loss.backward(retain_graph=True)
            with torch.no_grad():
                grad = perturbed_images.grad.data

                # 모멘텀 업데이트
                momentum = self.momentum_factor * momentum + grad / torch.norm(grad, p=1)  # L1 정규화

                # 마스킹된 영역과 비마스킹 영역에 각각 다른 강도로 그래디언트 적용
                mask_grad = momentum * mask_tensor * self.alpha  # 마스킹된 영역
                non_mask_grad = momentum * (1 - mask_tensor) * self.non_mask_alpha  # 비마스킹된 영역
                total_grad = mask_grad + non_mask_grad

                # Perturbation 업데이트
                perturbed_images = perturbed_images + total_grad.sign()

                # 마스킹된 영역과 비마스킹된 영역 각각의 노이즈 강도 제한
                perturbed_images = torch.clamp(
                    input_tensor + torch.clamp(perturbed_images - input_tensor, 
                                            -self.adversarial_budget, self.adversarial_budget) * mask_tensor +
                    torch.clamp(perturbed_images - input_tensor, 
                                -self.non_mask_budget, self.non_mask_budget) * (1 - mask_tensor),
                    0, 1
                ).detach_().requires_grad_(True)

        print(f"Max Loss Achieved: {max_loss:.4f}")

        # 최종 적대적 이미지 반환
        if best_adv_img is not None:
            adversarial_image = transforms.ToPILImage()(best_adv_img.squeeze().cpu())
        else:
            adversarial_image = transforms.ToPILImage()(perturbed_images.squeeze().cpu())
        return adversarial_image


# 5. Main
if __name__ == "__main__":
    # Set a good seed
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Load the model
    model_name = "runwayml/stable-diffusion-v1-5"
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
    attack = Img2ImgAdversarialAttack(
        model=model,
        vae=model.vae,
        tokenizer=model.tokenizer,
        noise_scheduler=model.scheduler,
        adversarial_budget=4/255,
        alpha=1/255,
        steps=100,
        momentum_factor=0.9,
        non_mask_budget=2/255,
        non_mask_alpha=0.5/255
    )
    start_time = time.time()
    adversarial_image = attack.generate_adversarial(input_image, prompt)
    end_time = time.time()

    print(f"Adversarial example generated in {end_time - start_time:.2f} seconds.")
    adversarial_image.show()
    adversarial_image.save("adversarial_image.png")

    # Generate outputs
    adversarial_output = model(prompt, image=adversarial_image, strength=0.6, guidance_scale=7).images[0]
    original_output= model(prompt, image=input_image, strength=0.6, guidance_scale=7).images[0]

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

    # 일단 주석 처리
    
    # Text-to-image example
    # txt2img_model = load_txt2img_model(model_name)
    # test_image = txt2img_model(prompt="person riding a bike", guidance_scale=5.5).images[0]
    # test_image.show()

    # Create output directory if it doesn't exist
    output_dir = "outputs"
    adversarial_dir = "adversarial"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(adversarial_dir, exist_ok=True)

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
            prompt = "portrait of a person with a neutral expression"

            start_time = time.time()
            adversarial_image = attack.generate_adversarial(input_image, prompt)
            end_time = time.time()

            adversarial_image.save(os.path.join(adversarial_dir, f"{base_filename}_adversarial_example.png"))
            print(f"Adversarial example generated in {end_time - start_time:.2f} seconds.")

            adversarial_output = model(prompt, image=adversarial_image, strength=0.6, guidance_scale=7).images[0]
            original_output = model(prompt, image=input_image, strength=0.6, guidance_scale=7).images[0]

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


            # Calculate and visualize noise
            noise_tensor = adversarial_tensor - input_tensor
            noise_tensor = (noise_tensor - noise_tensor.min()) / (noise_tensor.max() - noise_tensor.min())  # Normalize to [0, 1]
            noise_image = transforms.ToPILImage()(noise_tensor.squeeze().cpu())
            noise_image.save(os.path.join(adversarial_dir, f"{base_filename}_noise.png"))

            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([base_filename, lpips_original, lpips_adversarial, ssim_original, ssim_adversarial, psnr_original, psnr_adversarial])

            print(f"Processed {filename}: Metrics saved.")
            