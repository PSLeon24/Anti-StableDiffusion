import torch
from torch.nn.functional import kl_div, softmax, mse_loss
from diffusers import StableDiffusionImg2ImgPipeline
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import random
from torchvision.models import vgg16

# GPU 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# Stable Diffusion 모델 로드
def load_stable_diffusion_model(version="runwayml/stable-diffusion-v1-5"):
    return StableDiffusionImg2ImgPipeline.from_pretrained(version).to(device)

# 프롬프트 샘플링
PROMPT_TEMPLATES = [
    "A beautiful landscape with {adjective} mountains and rivers",
    "A photorealistic portrait of a {adjective} person",
    "A surreal {adjective} scene in a forest",
]
ADJECTIVES = ["beautiful", "futuristic", "vivid", "dreamlike", "realistic"]

def sample_prompt():
    template = random.choice(PROMPT_TEMPLATES)
    adjective = random.choice(ADJECTIVES)
    return template.format(adjective=adjective)

# VGG16 특징 추출기 초기화
feature_extractor = vgg16(pretrained=True).features[:16].to(device).eval()

# 특징 추출 함수
def extract_features(image):
    """
    이미지를 VGG16 특징으로 변환.
    """
    image = torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
    return feature_extractor(image)

# C&W 공격 함수
def carlini_wagner_attack(img, pipe, c=1.0, num_steps=100, lr=0.01):
    img_tensor = ToTensor()(img).unsqueeze(0).to(device).requires_grad_(True)
    adversarial_img = img_tensor.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([adversarial_img], lr=lr)

    for step in range(num_steps):
        # 랜덤 프롬프트 생성
        random_prompt = sample_prompt()
        print(f"Sampled Prompt: {random_prompt}")

        # 동일한 시드를 설정해 Stable Diffusion의 랜덤성을 제거
        generator = torch.Generator(device).manual_seed(42)

        # Stable Diffusion 실행
        original_output = pipe(prompt=random_prompt, image=img_tensor, strength=0.8, generator=generator).images[0]
        adversarial_output = pipe(prompt=random_prompt, image=adversarial_img, strength=0.8, generator=generator).images[0]

        # 텐서 변환
        original_tensor = ToTensor()(original_output).unsqueeze(0).to(device)
        adversarial_tensor = ToTensor()(adversarial_output).unsqueeze(0).to(device)

        # 특징 추출
        original_features = extract_features(original_tensor)
        adversarial_features = extract_features(adversarial_tensor)

        # 손실 계산
        # 1. MSE 손실: 원본 이미지와 적대적 이미지의 유사성 유지
        mse_loss_value = mse_loss(adversarial_img, img_tensor)

        # 2. KL Divergence 손실: 특징 분포 왜곡
        kl_loss_value = kl_div(
            softmax(original_features.flatten(), dim=0).log(),
            softmax(adversarial_features.flatten(), dim=0),
            reduction="batchmean"
        )

        # 전체 손실
        loss = c * kl_loss_value + mse_loss_value
        print(f"Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}")

        # 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 픽셀 값 클리핑
        adversarial_img.data = torch.clamp(adversarial_img, 0, 1)

    return adversarial_img

# 입력 이미지 로드
input_image_path = "Test/000001.jpg"
input_image = Image.open(input_image_path).convert("RGB")

# 모델 로드
model_version = "runwayml/stable-diffusion-v1-5"
pipe = load_stable_diffusion_model(version=model_version)

# C&W 공격 실행
adversarial_result = carlini_wagner_attack(input_image, pipe, c=10.0, num_steps=50, lr=0.005)

# 결과 저장
output_image = ToPILImage()(adversarial_result.squeeze(0).cpu())
output_image.save("adversarial_output_cw.jpg")
