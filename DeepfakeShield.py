# 1. Import Dependencies
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
from tqdm import tqdm

# 2. Set Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Load Stable Diffusion Model with img2img Pipeline from Hugging Face
def load_model(model_name="runwayml/stable-diffusion-v1-5"):
    """Load the Stable Diffusion img2img pipeline model."""
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe.safety_checker = None  # Disable the safety checker
    pipe = pipe.to(device)
    return pipe

# 4. Define Adversarial Attack on img2img Model
class Img2ImgAdversarialAttack:
    def __init__(self, model, epsilon=0.03, alpha=0.005, steps=50):
        self.model = model
        self.epsilon = epsilon  # Maximum perturbation
        self.alpha = alpha  # Step size
        self.steps = steps  # Number of steps

    def generate_adversarial(self, image, prompt):
        """Generates an adversarial image using img2img by maximizing the distortion from the original image."""
        # Prepare the input image in tensor format
        input_tensor = torch.tensor(np.array(image.resize((512, 512))) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        input_tensor = input_tensor.requires_grad_(True)

        for step in tqdm(range(self.steps), desc="Generating Adversarial Example"):
            # Generate output image from the img2img model
            output = self.model(prompt=prompt, image=input_tensor, strength=0.4, guidance_scale=5).images[0]

            # Resize the output image to match the input size, then convert to tensor
            output = output.resize((512, 512), Image.BICUBIC)
            output_tensor = torch.tensor(np.array(output) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

            # Calculate KL Divergence loss (maximize distortion)
            p = F.log_softmax(output_tensor, dim=-1)
            q = F.softmax(input_tensor, dim=-1)
            loss = F.kl_div(p, q, reduction="batchmean")

            loss.backward()

            # Apply gradient to input image
            grad_sign = input_tensor.grad.sign()
            input_tensor = input_tensor + self.alpha * grad_sign
            input_tensor = torch.clamp(input_tensor, min=0, max=1)

            # Ensure the perturbation does not exceed epsilon
            perturbation = torch.clamp(
                input_tensor - torch.tensor(np.array(image.resize((512, 512))) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device), min=-self.epsilon, max=self.epsilon)
            input_tensor = torch.tensor(np.array(image.resize((512, 512))) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) + perturbation
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
    epsilon = 0.03  # maximum noise level
    alpha = 0.005  # step size
    steps = 50  # number of attack iterations
    prompt = "portrait of a person with a neutral expression"

    # Initialize attack
    attack = Img2ImgAdversarialAttack(model=model, epsilon=epsilon, alpha=alpha, steps=steps)

    # Load input image and resize to 512x512 for stable processing
    input_image = Image.open("./Test/000001.jpg").convert("RGB").resize((512, 512))

    # Generate output for the original image
    original_output = model(prompt=prompt, image=input_image, strength=0.4, guidance_scale=5).images[0]
    original_output.save("original_output.png")
    original_output.show()

    # Generate adversarial example
    adversarial_image = attack.generate_adversarial(input_image, prompt)

    # Save and show the adversarial image
    adversarial_image.save("adversarial_image.png")
    adversarial_image.show()

    # Generate output for the adversarial image
    adversarial_output = model(prompt=prompt, image=adversarial_image, strength=0.4, guidance_scale=5).images[0]
    adversarial_output.save("adversarial_output.png")
    adversarial_output.show()
