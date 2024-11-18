# 1. Import Dependencies
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import time
from torchvision import transforms
import lpips  # LPIPS 패키지 설치 필요: pip install lpips

# 2. Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load Stable Diffusion Model with img2img Pipeline from Hugging Face
def load_model(model_name="runwayml/stable-diffusion-v1-5"):
    """Load the Stable Diffusion img2img pipeline model."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.safety_checker = None  # Disable the safety checker
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()  # 메모리 최적화
    return pipe

# 4. Define Adversarial Attack on img2img Model Using Perceptual Loss
class Img2ImgAdversarialAttack:
    def __init__(self, model, epsilon=0.2, alpha=0.01, steps=50, momentum_factor=0.9):
        self.model = model
        self.epsilon = epsilon  # 최대 섭동 크기
        self.alpha = alpha  # 스텝 크기
        self.steps = steps  # 반복 횟수
        self.momentum_factor = momentum_factor  # 모멘텀 계수
        self.momentum = None  # 모멘텀 초기화
        # LPIPS 손실 함수 초기화
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)

    def generate_adversarial(self, image, prompt):
        # 입력 이미지 텐서로 변환
        preprocess_input = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess_input(image).unsqueeze(0).to(device).float()
        original_image_tensor = input_tensor.clone().detach()

        # 초기 섭동 추가
        perturbation = torch.empty_like(input_tensor).uniform_(-self.epsilon, self.epsilon)
        input_tensor = input_tensor + perturbation
        input_tensor = torch.clamp(input_tensor, 0, 1)
        input_tensor.requires_grad = True

        self.momentum = torch.zeros_like(input_tensor)  # 모멘텀 초기화

        for step in tqdm(range(self.steps), desc="Generating Adversarial Example", leave=False):
            # 모델을 사용하여 출력 이미지 생성
            output_image = self.model(prompt=prompt, image=input_tensor, strength=0.4, guidance_scale=5).images[0]
            output_tensor = preprocess_input(output_image).unsqueeze(0).to(device).float()

            # LPIPS를 위한 텐서 전처리
            input_lpips = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
            output_lpips = F.interpolate(output_tensor, size=(256, 256), mode='bilinear', align_corners=False)

            # [-1, 1] 범위로 정규화
            input_lpips = (input_lpips - 0.5) / 0.5
            output_lpips = (output_lpips - 0.5) / 0.5

            # LPIPS 손실 계산
            loss = -self.loss_fn(output_lpips, input_lpips)
            loss = loss.mean()
            loss.backward()

            # 모멘텀 업데이트
            grad = input_tensor.grad
            self.momentum = self.momentum_factor * self.momentum + grad / (grad.norm() + 1e-8)

            # 섭동 적용
            input_tensor = input_tensor + self.alpha * self.momentum.sign()
            input_tensor = torch.clamp(input_tensor, 0, 1)

            # 섭동이 epsilon 내에 있도록 클리핑
            perturbation = torch.clamp(input_tensor - original_image_tensor, -self.epsilon, self.epsilon)
            input_tensor = original_image_tensor + perturbation
            input_tensor = input_tensor.detach().requires_grad_(True)

        # 최종 적대적 이미지 변환
        adversarial_image = transforms.ToPILImage()(input_tensor.squeeze().cpu())
        return adversarial_image

# 5. Main Execution
if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Load model
    model_name = "runwayml/stable-diffusion-v1-5"
    model = load_model(model_name)

    # Define attack parameters
    epsilon = 8/255  # Maximum noise level
    alpha = 0.05   # Step size
    steps = 100     # Number of attack iterations
    prompt = "portrait of a person with a neutral expression, purple hair"

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, epsilon=epsilon, alpha=alpha, steps=steps)

    # Load input image and resize to 512x512
    input_image = Image.open("./Test/000005.jpg").convert("RGB").resize((512, 512))

    # Generate adversarial example
    start_time = time.time()
    adversarial_image = attack.generate_adversarial(input_image, prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken to generate adversarial example: {elapsed_time:.2f} seconds")

    # Generate output for the adversarial image
    adversarial_output = model(prompt=prompt, image=adversarial_image, strength=0.4, guidance_scale=5).images[0]

    # Save and show the adversarial image
    adversarial_image.save("adversarial_image.png")
    adversarial_image.show()

    # Save and show the outputs
    adversarial_output.save("adversarial_output.png")
    adversarial_output.show()

    # Optional: Calculate and print LPIPS distance
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    adversarial_output_tensor = preprocess(adversarial_output).unsqueeze(0).to(device)

    # LPIPS distance between input image and adversarial output
    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_adversarial = lpips_loss_fn(input_tensor, adversarial_output_tensor).item()
    print(f"LPIPS distance between input image and adversarial output: {lpips_adversarial:.4f}")
