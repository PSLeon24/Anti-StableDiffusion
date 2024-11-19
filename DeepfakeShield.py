# 1. 필요한 라이브러리 임포트
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import lpips  # LPIPS: pip install lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torch.autograd import Variable
import random
import warnings
import time

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
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
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
    def __init__(self, model, adversarial_budget=16/255, alpha=1/255, steps=100, momentum_factor=0.9):
        self.model = model
        self.adversarial_budget = adversarial_budget
        self.alpha = alpha
        self.steps = steps
        self.momentum_factor = momentum_factor
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)

    def generate_adversarial(self, image, prompt):
        preprocess_input = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess_input(image).unsqueeze(0).to(device).float()
        original_image_tensor = input_tensor.clone().detach()
        adv_img = input_tensor.clone().detach().requires_grad_(True)
        self.momentum = torch.zeros_like(input_tensor)

        for step in tqdm(range(self.steps), desc="Generating adversarial example", leave=False):
            if adv_img.grad is not None:
                adv_img.grad.zero_()
            adv_img_ = (adv_img + 1.0) / 2.0
            adv_img_ = torch.clamp(adv_img_, 0.0, 1.0)

            # 이미지 생성
            output_image = self.model(prompt=prompt, image=adv_img_, strength=0.6, guidance_scale=7.5).images[0]
            output_tensor = preprocess_input(output_image).unsqueeze(0).to(device).float()

            # LPIPS 손실
            input_lpips = F.interpolate(adv_img, size=(256, 256), mode='bilinear', align_corners=False)
            output_lpips = F.interpolate(output_tensor, size=(256, 256), mode='bilinear', align_corners=False)

            input_lpips = (input_lpips - 0.5) / 0.5
            output_lpips = (output_lpips - 0.5) / 0.5

            lpips_loss = -self.loss_fn(output_lpips, input_lpips).mean()
            ssim_loss = 1 - ssim(output_tensor, original_image_tensor, data_range=1.0)
            brightness_loss = -torch.mean(torch.abs(output_tensor - original_image_tensor))

            # 최종 손실 계산
            loss = lpips_loss + ssim_loss + brightness_loss
            loss.backward()

            # 섭동 업데이트
            with torch.no_grad():
                grad = adv_img.grad.clone()  # NoneType 방지를 위해 grad 복사
                grad_accumulated = grad + self.momentum
                self.momentum = self.momentum_factor * self.momentum + grad_accumulated / (torch.norm(grad_accumulated) + 1e-8)
                adv_img = adv_img + self.alpha * self.momentum.sign()
                eta = torch.clamp(adv_img - original_image_tensor, -self.adversarial_budget, self.adversarial_budget)
                adv_img = torch.clamp(original_image_tensor + eta, 0, 1)
                adv_img.requires_grad_(True)  # grad 활성화

        adversarial_image_pil = transforms.ToPILImage()(adv_img.squeeze().cpu())
        return adversarial_image_pil

# 5. Main
if __name__ == "__main__":
    torch.manual_seed(42)

    # Load the model
    model_name = "stabilityai/stable-diffusion-2-1"
    model = load_model(model_name)

    # Load the input image
    input_image = Image.open("./Test/5.jpg").convert("RGB").resize((512, 512))
    prompt = "A high-quality portrait of a young person, highly detailed, realistic, smiling, with distinctive features."

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, adversarial_budget=16/255, alpha=1/255, steps=100)
    start_time = time.time()
    adversarial_image = attack.generate_adversarial(input_image, prompt)
    end_time = time.time()

    print(f"Adversarial example generated in {end_time - start_time:.2f} seconds.")
    adversarial_image.show()
    adversarial_image.save("adversarial_image.png")

    # Generate outputs
    adversarial_output = model(prompt=prompt, image=adversarial_image, strength=0.6, guidance_scale=7.5).images[0]
    original_output = model(prompt=prompt, image=input_image, strength=0.6, guidance_scale=7.5).images[0]

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

    print(f"Original LPIPS: {lpips_original:.4f}")
    print(f"Adversarial LPIPS: {lpips_adversarial:.4f}")
    print(f"Original SSIM: {ssim_original:.4f}")
    print(f"Adversarial SSIM: {ssim_adversarial:.4f}")
    print(f"Original PSNR: {psnr_original:.4f}")
    print(f"Adversarial PSNR: {psnr_adversarial:.4f}")

    # Text-to-image example
    txt2img_model = load_txt2img_model(model_name)
    test_image = txt2img_model(prompt="person riding a bike", guidance_scale=7).images[0]
    test_image.show()
