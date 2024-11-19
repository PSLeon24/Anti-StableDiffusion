import torch
from torch.nn.functional import kl_div, softmax
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import random
import os

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Stable Diffusion 모델 로드
def load_stable_diffusion_model(version="runwayml/stable-diffusion-v1-5"):
    """
    Stable Diffusion 모델 로드 함수.
    """
    return StableDiffusionImg2ImgPipeline.from_pretrained(version).to(device)

# CelebA-HQ 데이터셋에서 이미지 로드
def load_celeba_hq_images(data_dir, num_images=10):
    """
    CelebA-HQ 데이터셋에서 이미지를 로드.
    - data_dir: 데이터셋 디렉토리 경로
    - num_images: 로드할 이미지 수
    """
    image_paths = [os.path.join(data_dir, fname) for fname in os.listdir(data_dir) if fname.endswith(".jpg")]
    selected_images = random.sample(image_paths, num_images)
    return [Image.open(img_path).convert("RGB") for img_path in selected_images]

# 프롬프트 생성 함수
PROMPTS = [
    "A realistic portrait of a person with {adjective} features",
    "A photorealistic face with {adjective} expressions",
    "A highly detailed image of a person in {adjective} lighting"
]
ADJECTIVES = ["bright", "dramatic", "subtle", "soft", "natural", "vivid"]

def sample_prompt():
    """
    랜덤 프롬프트 생성.
    """
    template = random.choice(PROMPTS)
    adjective = random.choice(ADJECTIVES)
    return template.format(adjective=adjective)

# 빠르고 강력한 PGD 공격
def fast_pgd_attack(img, pipe, num_steps=5, epsilon=0.1, alpha=0.05):
    """
    빠른 PGD 공격으로 적대적 예제를 생성.
    - img: 원본 이미지 (PIL Image)
    - pipe: Stable Diffusion Img2Img Pipeline 객체
    - num_steps: 공격 단계 수
    - epsilon: 최대 perturbation 크기
    - alpha: 각 업데이트 크기
    """
    img_tensor = ToTensor()(img).unsqueeze(0).to(device).requires_grad_(True)
    adv_img = img_tensor.clone().detach()

    for step in range(num_steps):
        # 랜덤 프롬프트 생성
        prompt = sample_prompt()
        print(f"Sampled Prompt: {prompt}")

        # Stable Diffusion 실행
        with torch.no_grad():
            output_img = pipe(prompt=prompt, image=adv_img, strength=0.8).images[0]
            output_tensor = ToTensor()(output_img).unsqueeze(0).to(device)

        # KL Divergence 손실 계산
        loss = kl_div(
            softmax(img_tensor.flatten(), dim=0).log(),
            softmax(output_tensor.flatten(), dim=0),
            reduction="batchmean"
        )

        # Gradient 계산
        loss.backward()

        # PGD 업데이트
        adv_img = adv_img + alpha * adv_img.grad.sign()
        delta = torch.clamp(adv_img - img_tensor, min=-epsilon, max=epsilon)
        adv_img = torch.clamp(img_tensor + delta, min=0, max=1).detach().requires_grad_(True)

        print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

    return adv_img

# 메인 실행 코드
def main(data_dir, output_dir, num_images=10):
    """
    CelebA-HQ 데이터셋을 대상으로 적대적 예제를 생성.
    - data_dir: CelebA-HQ 데이터셋 디렉토리
    - output_dir: 생성된 적대적 예제를 저장할 디렉토리
    - num_images: 처리할 이미지 수
    """
    # 데이터 로드
    images = load_celeba_hq_images(data_dir, num_images=num_images)

    # 모델 로드
    model_version = "runwayml/stable-diffusion-v1-5"
    pipe = load_stable_diffusion_model(version=model_version)

    # 결과 저장 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 적대적 예제 생성 및 저장
    for idx, img in enumerate(images):
        print(f"Processing image {idx + 1}/{num_images}")
        adv_img = fast_pgd_attack(img, pipe, num_steps=10, epsilon=0.1, alpha=0.02)
        output_path = os.path.join(output_dir, f"adversarial_image_{idx + 1}.jpg")
        ToPILImage()(adv_img.squeeze(0).cpu()).save(output_path)

    print(f"Generated adversarial examples saved in {output_dir}")

# 실행
if __name__ == "__main__":
    data_dir = "Test/"  # CelebA-HQ 데이터셋 디렉토리
    output_dir = "./"   # 결과 저장 디렉토리
    num_images = 10                 # 처리할 이미지 수

    main(data_dir, output_dir, num_images=num_images)
