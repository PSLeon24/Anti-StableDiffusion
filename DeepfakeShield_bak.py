저코드는 동작이 안되니까 아래코드에 수정해줘

# 1. Import Dependencies
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import time

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

# 4. Define PGD-based Adversarial Attack on img2img Model
class Img2ImgAdversarialAttack:
    def __init__(self, model, epsilon=0.05, alpha=0.005, steps=50, alpha_min=0.0005, alpha_max=0.02,
                 momentum_factor=0.05):
        self.model = model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Initial step size
        self.steps = steps  # Number of steps
        self.alpha_min = alpha_min  # Minimum adaptive alpha
        self.alpha_max = alpha_max  # Maximum adaptive alpha
        self.momentum_factor = momentum_factor  # Momentum factor for gradient update
        self.momentum = None  # Initialize momentum as None

    def generate_adversarial(self, image, prompt):
        """Generates an adversarial image using img2img by maximizing the distortion from the original image using PGD with momentum."""
        # Prepare the input image in tensor format
        input_tensor = torch.tensor(np.array(image.resize((512, 512))) / 255.0, dtype=torch.float16).permute(2, 0,
                                                                                                             1).unsqueeze(
            0).to(device)
        original_image_tensor = input_tensor.clone()  # Store original image tensor for epsilon bound check

        # Add initial random noise within epsilon bounds
        input_tensor = input_tensor + torch.empty_like(input_tensor).uniform_(-self.epsilon, self.epsilon)
        input_tensor = torch.clamp(input_tensor, 0, 1)  # Ensure values are within valid range
        input_tensor = input_tensor.requires_grad_(True)

        self.momentum = torch.zeros_like(input_tensor)  # Initialize momentum to zero

        for step in tqdm(range(self.steps), desc="Generating Adversarial Example", leave=False):
            # Generate output image from the img2img model
            output = self.model(prompt=prompt, image=input_tensor, strength=0.4, guidance_scale=5).images[0]
            output = output.resize((512, 512), Image.BICUBIC)
            output_tensor = torch.tensor(np.array(output) / 255.0, dtype=torch.float16).permute(2, 0, 1).unsqueeze(
                0).to(device)

            # Calculate KL Divergence loss
            p = F.log_softmax(output_tensor, dim=-1)
            q = F.softmax(input_tensor, dim=-1)
            loss = F.kl_div(p, q, reduction="batchmean")
            loss.backward()

            # Calculate gradient and apply momentum
            grad = input_tensor.grad
            momentum_update = self.momentum_factor * self.momentum + grad / grad.norm()

            # Check if momentum direction is inconsistent with gradient direction
            if (momentum_update * grad).sum() < 0:  # Negative inner product implies opposite direction
                self.momentum = torch.zeros_like(input_tensor)
                print("Resetting momentum due to inconsistent direction.")
            else:
                self.momentum = momentum_update
                print(f"Step {step + 1}, Momentum Norm: {self.momentum.norm().item():.4f}")

            # Adaptive alpha: Adjust alpha based on gradient magnitude
            grad_norm = grad.norm()
            if grad_norm > 1e-6:  # To prevent division by zero
                self.alpha = min(max(self.alpha * (1.0 / grad_norm), self.alpha_min), self.alpha_max)

            # Apply PGD update rule with momentum and clipping
            input_tensor = input_tensor + self.alpha * self.momentum.sign()
            input_tensor = torch.clamp(input_tensor, min=0, max=1)

            # Ensure perturbation remains within epsilon bounds ~ L-infinity Norm
            perturbation = torch.clamp(input_tensor - original_image_tensor, min=-self.epsilon, max=self.epsilon)
            input_tensor = original_image_tensor + perturbation

            input_tensor = input_tensor.detach().requires_grad_(True)

        # Convert final adversarial image tensor to PIL format
        adversarial_image = (input_tensor.squeeze().permute(1, 2, 0).detach() * 255).cpu().numpy().astype(np.uint8)
        adversarial_image = Image.fromarray(adversarial_image)
        return adversarial_image


# 5. Main Execution
if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Load model and tokenizer
    model_name = "runwayml/stable-diffusion-v1-5"
    model = load_model(model_name)

    # Define attack parameters
    epsilon = 30/255  # maximum noise level
    alpha = 0.001  # step size
    steps = 20  # number of attack iterations
    prompt = "portrait of a person with a neutral expression, purple hair"
    # another prompt
    # "portrait of a person with a neutral expression"
    # "A high-quality image of a human face"

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, epsilon=epsilon, alpha=alpha, steps=steps)

    # Load input image and resize to 512x512 for stable processing
    input_image = Image.open("./Test/000005.jpg").convert("RGB").resize((512, 512))

    # Generate output for the original image
    original_output = model(prompt=prompt, image=input_image, strength=0.4, guidance_scale=5).images[0]
    original_output_tensor = torch.tensor(np.array(original_output.resize((512, 512))) / 255.0,
                                          dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

    # Generate adversarial example
    # Measure time for generating adversarial example
    start_time = time.time()
    adversarial_image = attack.generate_adversarial(input_image, prompt)
    end_time = time.time()

    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to generate adversarial example: {elapsed_time:.2f} seconds")

    # Generate output for the adversarial image
    adversarial_output = model(prompt=prompt, image=adversarial_image, strength=0.4, guidance_scale=5).images[0]
    adversarial_output_tensor = torch.tensor(np.array(adversarial_output.resize((512, 512))) / 255.0,
                                             dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

    # Save and show the adversarial image
    adversarial_image.save("adversarial_image.png")
    adversarial_image.show()

    # Save and show the outputs
    original_output.save("original_output.png")
    original_output.show()
    adversarial_output.save("adversarial_output.png")
    adversarial_output.show()

    # Calculate and print KL Divergence for two cases
    # Case 1: Input image and its output
    p1 = F.log_softmax(original_output_tensor, dim=-1)
    q1 = F.softmax(
        torch.tensor(np.array(input_image) / 255.0, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device),
        dim=-1)
    kl_div_original = F.kl_div(p1, q1, reduction="batchmean")
    print("KL Divergence between input image and its output:", kl_div_original.item())

    # Case 2: Input image and adversarial output
    p2 = F.log_softmax(adversarial_output_tensor, dim=-1)
    q2 = F.softmax(
        torch.tensor(np.array(input_image) / 255.0, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device),
        dim=-1)
    kl_div_adversarial = F.kl_div(p2, q2, reduction="batchmean")
    print("KL Divergence between input image and adversarial output:", kl_div_adversarial.item())

    # Iterate over images from 000002.jpg to 000100.jpg
    for i in range(2, 101):
        img_path = f"./Test/{i:06d}.jpg"
        try:
            # Load input image and resize to 512x512
            input_image = Image.open(img_path).convert("RGB").resize((512, 512))

            # Generate output for the original image
            original_output = model(prompt=prompt, image=input_image, strength=0.4, guidance_scale=5).images[0]
            original_output_tensor = torch.tensor(np.array(original_output.resize((512, 512))) / 255.0,
                                                  dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

            # Generate adversarial example
            # Measure time for generating adversarial example
            start_time = time.time()
            adversarial_image = attack.generate_adversarial(input_image, prompt)
            end_time = time.time()

            elapsed_time = end_time - start_time
            print(f"Time taken for Image {i:06d}: {elapsed_time:.2f} seconds")

            # Generate output for the adversarial image
            adversarial_output = model(prompt=prompt, image=adversarial_image, strength=0.4, guidance_scale=5).images[0]
            adversarial_output_tensor = torch.tensor(np.array(adversarial_output.resize((512, 512))) / 255.0,
                                                     dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

            # Calculate KL Divergence for two cases
            # Case 1: Input image and its output
            p1 = F.log_softmax(original_output_tensor, dim=-1)
            q1 = F.softmax(
                torch.tensor(np.array(input_image) / 255.0, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(
                    device), dim=-1)
            kl_div_original = F.kl_div(p1, q1, reduction="batchmean")

            # Case 2: Input image and adversarial output
            p2 = F.log_softmax(adversarial_output_tensor, dim=-1)
            q2 = F.softmax(
                torch.tensor(np.array(input_image) / 255.0, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(
                    device), dim=-1)
            kl_div_adversarial = F.kl_div(p2, q2, reduction="batchmean")

            # Print KL Divergence results for each image
            print(
                f"Image {i:06d}: KL Divergence (Original) = {kl_div_original.item():.4f}, KL Divergence (Adversarial) = {kl_div_adversarial.item():.4f}")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")