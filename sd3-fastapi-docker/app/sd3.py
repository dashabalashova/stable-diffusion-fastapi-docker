import torch
from diffusers import StableDiffusion3Pipeline

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} ...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,        # убираем safety checker
).to(device)

def generate_image(prompt: str, num_inference_steps: int = 4):
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0       # turbo без CFG
    ).images[0]
    return image
