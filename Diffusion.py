from diffusers import StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os
from PIL import Image

# Set the Hugging Face token
os.environ["HF_TOKEN"] = "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"

# Load the pre-trained Stable Diffusion model version 2.1
model_id = "stabilityai/stable-diffusion-2-1"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, use_auth_token=True).to("cuda")

# Load the tokenizer and text encoder
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", use_auth_token=True)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", use_auth_token=True).to("cuda")

# Define the function to generate images
def generate_images(prompt, num_images=1, guidance_scale=7.5):
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    # Encode the text
    text_embeddings = text_encoder(**inputs).last_hidden_state
    
    # Generate images
    images = pipe(prompt=prompt, num_images_per_prompt=num_images, guidance_scale=guidance_scale).images
    
    return images

# Example usage
prompt = "A fantasy landscape with mountains and a river"
images = generate_images(prompt, num_images=3)

# Save the generated images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")

# Define a function to preprocess the dataset
def preprocess_function(examples):
    # Use the images directly if they are already loaded
    images = [img.convert("RGB") for img in examples["image"]]
    captions = examples["caption"]
    return {"images": images, "captions": captions}

# Load your dataset
dataset = load_dataset("imagefolder", data_dir="data/class")

# Preprocess the dataset
dataset = dataset.map(preprocess_function, batched=True)

# Define the training arguments
training_args = TrainingArguments(
    output_dir="output_dir",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    fp16=True,
    save_steps=10,
    save_total_limit=2,
    remove_unused_columns=False,
    push_to_hub=False,
)

# Initialize the trainer
trainer = Trainer(
    model=pipe,
    args=training_args,
    train_dataset=dataset["train"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("dreambooth_fine_tuned")

# Load the fine-tuned model
fine_tuned_pipe = StableDiffusionPipeline.from_pretrained("dreambooth_fine_tuned", torch_dtype=torch.float16).to("cuda")

# Define the function to generate images using the fine-tuned model
def generate_images(prompt, num_images=1, guidance_scale=7.5):
    # Generate images
    images = fine_tuned_pipe(prompt=prompt, num_images_per_prompt=num_images, guidance_scale=guidance_scale).images
    return images

# Example usage with fine-tuned model
prompt = "A fantasy landscape with mountains and a river"
images = generate_images(prompt, num_images=3)

# Save the generated images
for i, img in enumerate(images):
    img.save(f"generated_image_{i}.png")