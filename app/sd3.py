import torch
from diffusers import StableDiffusion3Pipeline
import time

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} ...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def generate_image(prompt: str, num_inference_steps: int = 4):
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0
    ).images[0]
    return image

def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

def generate_images(prompts: list[str], chunk_size: int = 4, num_inference_steps: int = 4):
    results = []
    for sub_prompts in chunked(prompts, chunk_size):
        images = pipe(
            sub_prompts,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        ).images
        results.extend(images)
        torch.cuda.empty_cache()
    return results
