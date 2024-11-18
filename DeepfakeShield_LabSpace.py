# 1. Import Dependencies
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm
import time
from skimage import color # LAB Space

# 2. Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"


# 3. Load Stable Diffusion Model with img2img Pipeline from Hugging Face
def load_model(model_name="runwayml/stable-diffusion-v1-5"):
    """Load the Stable Diffusion img2img pipeline model."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.safety_checker = None  # Disable the safety checker
    pipe = pipe.to(device)
    return pipe

# 4. Define PGD-based Adversarial Attack on img2img Model with L-channel Focused Perturbation
# 4. Define PGD-based Adversarial Attack on img2img Model with L-channel Focused Perturbation
class Img2ImgAdversarialAttack:
    def __init__(self, model, epsilon=0.05, alpha=0.005, steps=50, alpha_min=0.001, alpha_max=0.01,
                 momentum_factor=0.9):
        self.model = model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Initial step size
        self.steps = steps  # Number of steps
        self.alpha_min = alpha_min  # Minimum adaptive alpha
        self.alpha_max = alpha_max  # Maximum adaptive alpha
        self.momentum_factor = momentum_factor  # Momentum factor for gradient update
        self.momentum = None  # Initialize momentum as None

    def generate_adversarial(self, image, prompt):
        """Generates an adversarial image using img2img by maximizing the distortion in L channel from the original image using PGD with momentum in Lab space."""

        # Convert image to Lab color space
        lab_image = color.rgb2lab(np.array(image.resize((512, 512))) / 255.0)
        input_tensor = torch.tensor(lab_image, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

        # Add initial random noise only to the L channel within epsilon bounds
        input_tensor = input_tensor.clone()  # Prevent in-place operations on requires_grad=True tensor
        input_tensor[0, 0, :, :] += torch.empty_like(input_tensor[0, 0, :, :]).uniform_(-self.epsilon, self.epsilon)
        input_tensor = input_tensor.requires_grad_(True)

        self.momentum = torch.zeros_like(input_tensor)  # Initialize momentum to zero

        for step in tqdm(range(self.steps), desc="Generating Adversarial Example", leave=False):
            # Convert Lab tensor to RGB for model processing
            rgb_input_tensor = color.lab2rgb(input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy())
            rgb_input_tensor = torch.tensor(rgb_input_tensor, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

            # Generate output image from the img2img model
            output = self.model(prompt=prompt, image=rgb_input_tensor, strength=0.4, guidance_scale=5).images[0]
            output = output.resize((512, 512), Image.BICUBIC)
            output_tensor = torch.tensor(np.array(output) / 255.0, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device)

            # Calculate KL Divergence loss
            p = F.log_softmax(output_tensor, dim=-1)
            q = F.softmax(input_tensor, dim=-1)
            loss = F.kl_div(p, q, reduction="batchmean")
            loss.backward()

            # Calculate gradient and apply momentum
            grad = input_tensor.grad
            self.momentum = self.momentum_factor * self.momentum + grad / (grad.norm() + 1e-6)  # Update momentum to avoid NaN
            print(f"Step {step + 1}, Momentum Norm: {self.momentum.norm().item():.4f}")  # Print momentum norm

            # Adaptive alpha: Adjust alpha based on gradient magnitude for L channel
            grad_norm = grad[0, 0, :, :].norm()
            if grad_norm > 1e-6:  # To prevent division by zero
                self.alpha = min(max(self.alpha * (1.0 / grad_norm), self.alpha_min), self.alpha_max)

            # Apply PGD update rule with momentum and clipping on L channel only
            input_tensor = input_tensor.clone()  # Clone to prevent in-place operations
            input_tensor[0, 0, :, :] = input_tensor[0, 0, :, :] + self.alpha * self.momentum[0, 0, :, :].sign()
            input_tensor[0, 0, :, :] = torch.clamp(input_tensor[0, 0, :, :], min=-self.epsilon, max=self.epsilon)

            # Ensure perturbation remains within epsilon bounds for L channel
            perturbation = torch.clamp(
                input_tensor - torch.tensor(lab_image, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device),
                min=-self.epsilon, max=self.epsilon
            )
            input_tensor = torch.tensor(lab_image, dtype=torch.float16).permute(2, 0, 1).unsqueeze(0).to(device) + perturbation
            input_tensor = input_tensor.detach().requires_grad_(True)

        # Convert final adversarial Lab tensor back to RGB and then to PIL format
        adversarial_lab_image = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        adversarial_rgb_image = color.lab2rgb(adversarial_lab_image) * 255
        adversarial_image = Image.fromarray(adversarial_rgb_image.astype(np.uint8))
        return adversarial_image




# 5. Main Execution
if __name__ == "__main__":
    # Set the random seed for reproducibility
    torch.manual_seed(42)

    # Load model and tokenizer
    model_name = "runwayml/stable-diffusion-v1-5"
    model = load_model(model_name)

    # Define attack parameters
    epsilon = 0.4  # maximum noise level
    alpha = 0.01  # step size
    steps = 50  # number of attack iterations
    prompt = "portrait of a person with a neutral expression"
    # another prompt
    # "A high-quality image of a human face"

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, epsilon=epsilon, alpha=alpha, steps=steps)

    # Load input image and resize to 512x512 for stable processing
    input_image = Image.open("./Test/000001.jpg").convert("RGB").resize((512, 512))

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
