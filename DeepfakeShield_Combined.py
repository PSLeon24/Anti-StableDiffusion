# 1. Import Dependencies
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import time
from torchvision import transforms
import lpips  # LPIPS 패키지 설치 필요: pip install lpips
from torchmetrics.functional import structural_similarity_index_measure as ssim

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

# 4. Define Adversarial Attack on img2img Model Using Combined Losses
class Img2ImgAdversarialAttack:
    def __init__(self, model, epsilon=0.2, alpha=0.01, steps=50, momentum_factor=0.9):
        self.model = model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Step size
        self.steps = steps  # Number of attack iterations
        self.momentum_factor = momentum_factor  # Momentum factor
        self.momentum = None  # Initialize momentum
        # LPIPS loss function initialization
        self.loss_fn = lpips.LPIPS(net='vgg').to(device)

    def generate_adversarial(self, image, prompt):
        # Preprocess input image
        preprocess_input = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        input_tensor = preprocess_input(image).unsqueeze(0).to(device).float()
        original_image_tensor = input_tensor.clone().detach()

        # Initialize perturbation
        perturbation = torch.empty_like(input_tensor).uniform_(-self.epsilon, self.epsilon)
        input_tensor = input_tensor + perturbation
        input_tensor = torch.clamp(input_tensor, 0, 1)
        input_tensor.requires_grad = True

        self.momentum = torch.zeros_like(input_tensor)  # Initialize momentum

        for step in tqdm(range(self.steps), desc="Generating Adversarial Example", leave=False):
            # Generate output image
            output_image = self.model(prompt=prompt, image=input_tensor, strength=0.4, guidance_scale=5).images[0]
            output_tensor = preprocess_input(output_image).unsqueeze(0).to(device).float()

            # LPIPS loss preprocessing
            input_lpips = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
            output_lpips = F.interpolate(output_tensor, size=(256, 256), mode='bilinear', align_corners=False)

            # Normalize to [-1, 1]
            input_lpips = (input_lpips - 0.5) / 0.5
            output_lpips = (output_lpips - 0.5) / 0.5

            # Calculate LPIPS loss
            lpips_loss = -self.loss_fn(output_lpips, input_lpips).mean()

            # Calculate MSE loss
            #mse_loss = F.mse_loss(output_tensor, original_image_tensor)

            # Calculate SSIM loss
            ssim_loss = 1 - ssim(output_tensor, original_image_tensor, data_range=1.0)

            # Combined loss
            loss = lpips_loss # + 0.4 * ssim_loss # + mse_loss
            loss.backward()

            # Update momentum and apply perturbation
            grad = input_tensor.grad
            self.momentum = self.momentum_factor * self.momentum + grad / (grad.norm() + 1e-8) # 0으로 나뉘는걸 막기 위해서
            input_tensor = input_tensor + self.alpha * self.momentum.sign()
            input_tensor = torch.clamp(input_tensor, 0, 1)

            # Ensure perturbation is within epsilon bounds
            perturbation = torch.clamp(input_tensor - original_image_tensor, -self.epsilon, self.epsilon)
            input_tensor = original_image_tensor + perturbation
            input_tensor = input_tensor.detach().requires_grad_(True)

        # Convert final adversarial image tensor to PIL format
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
    epsilon = 2 / 255  # Maximum noise level
    alpha = 0.1  # Step size
    steps = 10  # Number of attack iterations
    prompt = "portrait of a person with a neutral expression, purple hair"

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, epsilon=epsilon, alpha=alpha, steps=steps)

    # Load input image and resize to 512x512
    input_image = Image.open("./Test/000001.jpg").convert("RGB").resize((512, 512))

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

    # Optional: Print Variance of the output image
    output_tensor = transforms.ToTensor()(adversarial_output).unsqueeze(0).to(device).float()
    variance = torch.var(output_tensor).item()
    print(f"Variance of the adversarial output: {variance:.4f}")

    # LPIPS distance between input image and adversarial output
    # Optional: Calculate and print LPIPS distance
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    adversarial_output_tensor = preprocess(adversarial_output).unsqueeze(0).to(device)

    lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)
    lpips_adversarial = lpips_loss_fn(input_tensor, adversarial_output_tensor).item()
    print(f"LPIPS distance between input image and adversarial output: {lpips_adversarial:.4f}")