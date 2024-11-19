# 1. 필요한 라이브러리 임포트
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import time
from torchvision import transforms
import lpips  # LPIPS 패키지 설치 필요: pip install lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torch.autograd import Variable
import numpy as np
import random

# 2. 디바이스 설정
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Hugging Face에서 img2img 파이프라인을 사용한 Stable Diffusion 모델 로드
def load_model(model_name="runwayml/stable-diffusion-v1-5"):
    """Stable Diffusion img2img 파이프라인 모델 로드."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.safety_checker = None  # 안전 체크 비활성화
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # 메모리 최적화
    return pipe

# 4. img2img 모델에 대한 적대적 공격 정의 (Shadow Attack 기반 기법 사용)
class Img2ImgAdversarialAttack:
    def __init__(self, model, epsilon=0.2, alpha=0.01, steps=50, momentum_factor=0.9):
        self.model = model
        self.epsilon = epsilon  # 최대 섭동량
        self.alpha = alpha  # 스텝 크기
        self.steps = steps  # 공격 반복 횟수
        self.momentum_factor = momentum_factor  # 모멘텀 계수
        self.momentum = None  # 모멘텀 초기화
        # LPIPS 손실 함수 초기화
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)

    def generate_adversarial(self, image, prompt):
        # 입력 이미지 전처리
        preprocess_input = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess_input(image).unsqueeze(0).to(device).float()
        original_image_tensor = input_tensor.clone().detach()

        # 섭동 초기화
        random_init = torch.rand_like(input_tensor) * 0.003
        adv_img = Variable(torch.clamp(input_tensor.data + random_init.data, 0, 1), requires_grad=True)

        # Shadow Attack 기반 공격 수행
        for step in tqdm(range(self.steps), desc="적대적 예제 생성 중", leave=False):
            if step % 10 != 0:
                continue
            if adv_img.grad is not None:
                adv_img.grad.zero_()
            adv_img_ = torch.clamp((adv_img + 1.0) / 2.0, min=0.0, max=1.0)

            # 모델 출력 계산
            output_image = self.model(prompt=prompt, image=adv_img_, strength=0.4, guidance_scale=5).images[0]
            output_tensor = preprocess_input(output_image).unsqueeze(0).to(device).float()

            # LPIPS 손실 전처리
            input_lpips = F.interpolate(adv_img, size=(256, 256), mode='bilinear', align_corners=False)
            output_lpips = F.interpolate(output_tensor, size=(256, 256), mode='bilinear', align_corners=False)

            # [-1, 1] 범위로 정규화
            input_lpips = (input_lpips - 0.5) / 0.5
            output_lpips = (output_lpips - 0.5) / 0.5

            # LPIPS 손실 계산
            lpips_loss = -self.loss_fn(output_lpips, input_lpips).mean()

            # SSIM 손실 계산
            ssim_loss = 1 - ssim(output_tensor, original_image_tensor, data_range=1.0)

            # 결합 손실 계산
            shadow_factor = random.uniform(0.8, 1.2)  # Shadow 공격 요소로 임의의 변동성 추가
            loss = shadow_factor * (1000 * lpips_loss + 1000 * ssim_loss)
            loss.backward()

            # 섭동 업데이트
            with torch.no_grad():
                grad_info = self.alpha * adv_img.grad.data.sign()
                adv_img = adv_img.data + grad_info
                eta = torch.clamp(adv_img.data - original_image_tensor.data, -self.epsilon, self.epsilon)
                adv_img = original_image_tensor.data + eta

            adv_img = Variable(torch.clamp(adv_img, 0, 1), requires_grad=True)

        # 최종 적대적 이미지 텐서를 PIL 형식으로 변환
        adversarial_image = transforms.ToPILImage()(adv_img.squeeze().cpu())
        return adversarial_image


# 5. 메인 실행
if __name__ == "__main__":
    # 재현성을 위한 랜덤 시드 설정
    torch.manual_seed(42)

    # 모델 로드
    model_name = "runwayml/stable-diffusion-v1-5"
    model = load_model(model_name)

    # 공격 파라미터 정의
    epsilon = 0.1  # 최대 노이즈 레벨
    alpha = 0.02  # 스텝 크기
    steps = 100  # 공격 반복 횟수
    prompt = "portrait of a person with a neutral expression, purple hair"

    # 공격 초기화
    attack = Img2ImgAdversarialAttack(model=model, epsilon=epsilon, alpha=alpha, steps=steps)

    # 입력 이미지 로드 및 크기 조정 (512x512)
    input_image = Image.open("./Test/000001.jpg").convert("RGB").resize((512, 512))

    # 적대적 예제 생성
    start_time = time.time()
    adversarial_image = attack.generate_adversarial(input_image, prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"적대적 예제 생성에 소요된 시간: {elapsed_time:.2f} 초")

    # 적대적 이미지에 대한 출력 생성
    adversarial_output = model(prompt=prompt, image=adversarial_image, strength=0.4, guidance_scale=5).images[0]

    # 적대적 이미지 저장 및 표시
    adversarial_image.save("adversarial_image.png")
    adversarial_image.show()

    # 출력 이미지 저장 및 표시
    adversarial_output.save("adversarial_output.png")
    adversarial_output.show()

    # 선택 사항: 출력 이미지의 분산 출력
    output_tensor = transforms.ToTensor()(adversarial_output).unsqueeze(0).to(device).float()
    variance = torch.var(output_tensor).item()
    print(f"적대적 출력의 분산: {variance:.4f}")

    # 입력 이미지와 적대적 출력 사이의 LPIPS 거리
    # 선택 사항: LPIPS 거리 계산 및 출력
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    adversarial_output_tensor = preprocess(adversarial_output).unsqueeze(0).to(device)

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_adversarial = lpips_loss_fn(input_tensor, adversarial_output_tensor).item()
    print(f"입력 이미지와 적대적 출력 사이의 LPIPS 거리: {lpips_adversarial:.4f}")
