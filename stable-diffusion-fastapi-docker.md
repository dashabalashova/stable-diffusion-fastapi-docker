# Stable Diffusion 3.5 Turbo – FastAPI + Docker Deployment Guide

This tutorial explains how to deploy **Stable Diffusion 3.5 Turbo** as a **FastAPI service** inside **Docker** on a GPU-enabled VM (Nebius Compute in this example).

![Dragon](images/dragon.png)

---

## 1. Create GPU VM

1. Open [Nebius Console](https://console.nebius.com/compute).  
2. Click **Create resource** → **Virtual machine** (top right).  
3. Fill in the fields:  
   - **Name**: `stable-diffusion`  
   - **Platform**: *NVIDIA® L40S PCIe with Intel Ice Lake (1 GPU)*  
4. Add **credentials** (username + SSH key):  
   - On your local machine, generate a new SSH key pair:  
     ```bash
     ssh-keygen -t ed25519 -C "your_email@example.com"
     ```
     Press **Enter** to accept defaults. This creates:  
     - `~/.ssh/id_ed25519` (private key)  
     - `~/.ssh/id_ed25519.pub` (public key)  
   - Show the public key:  
     ```bash
     cat ~/.ssh/id_ed25519.pub
     ```
   - In Nebius Console → **Add credentials**:  
     - **Username**: choose your login name  
     - **Public key**: paste the contents of `id_ed25519.pub`  
5. Finish VM creation.  
6. In [Nebius Console](https://console.nebius.com/compute):  
   - Open the **Compute** section.  
   - Find your created VM in the list (look for the name `stable-diffusion`).  
   - Click on the VM to open its details.  
   - Locate the **Public IPv4** field and copy the IP address (you will use it to connect via SSH).  

---

## 2. Verify Environment and Setup Project

Connect to your VM via SSH:  
```bash
ssh <Username>@<Public-IP> 
```

Check Python and CUDA:

```bash
python3 --version
nvcc --version
```

Expected output:

- Python 3.12+
- CUDA 12.8+

Download the project:

```bash
git clone https://github.com/dashabalashova/stable-diffusion-fastapi-docker.git
cd stable-diffusion-fastapi-docker
```

---

## 3. Project Files

### Dockerfile

```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

# enable hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

COPY app ./app

EXPOSE 8000 7860

# By default we launch FastAPI, but this can be overridden
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### requirements.txt

```txt
# Hugging Face stack
diffusers>=0.31.0
transformers>=4.44.0
accelerate>=0.33.0
safetensors>=0.4.3
sentencepiece
protobuf==4.25.3

# Vision
torchvision==0.23.0

# API
fastapi
uvicorn[standard]
pillow
hf_transfer

# Web UI
gradio
```

---

### app/sd3.py

```python
import torch
from diffusers import StableDiffusion3Pipeline

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
```

---

### app/main.py

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from io import BytesIO
from .sd3 import generate_image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Stable Diffusion 3.5 Turbo API is running"}

@app.get("/generate")
def generate(prompt: str):
    image = generate_image(prompt)
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
```

---

### app/webui.py

```python
import gradio as gr
from .sd3 import generate_image

def infer(prompt):
    image = generate_image(prompt)
    return image

def launch_gradio():
    demo = gr.Interface(
        fn=infer,
        inputs=gr.Textbox(label="Prompt", placeholder="A small dragon drinking coffee in Amsterdam, watercolor"),
        outputs=gr.Image(type="pil"),
        title="Stable Diffusion 3.5 Turbo WebUI",
        description="Enter a prompt and generate an image with Stable Diffusion 3.5 Turbo"
    )
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_gradio()
```

---

### client.py

```python
import argparse
import requests

def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion 3.5 Turbo client")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation")
    parser.add_argument("--host", type=str, default="localhost", help="API host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument("--output", type=str, default="output.png", help="Output image filename")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}/generate"
    print(f"➡️ Sending request to {api_url} ...")

    response = requests.get(api_url, params={"prompt": args.prompt})
    if response.status_code == 200:
        with open(args.output, "wb") as f:
            f.write(response.content)
        print(f"✅ Image saved as {args.output}")
    else:
        print("❌ Error:", response.status_code, response.text)

if __name__ == "__main__":
    main()
```

---

## 4. Build Docker Image

Enable Docker for your user: 

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Build image:

```bash
docker build -t sd3 .
```

---

## 5. Run Service (Options)

### Run API (FastAPI)

```bash
docker run --gpus all -p 8000:8000 sd3 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run WebUI (Gradio)

```bash
docker run --gpus all -p 7860:7860 sd3 python -m app.webui
```

---

## 6. Test API

### Curl (from VM)

```bash
curl "http://localhost:8000/generate?prompt=a+small+friendly+dragon+reading+a+bedtime+story+to+forest+animals,+soft+watercolor+textures,+pastel+palette,+rounded+shapes,+whimsical+illustration,+light+paper+grain,+heartwarming+mood" --output dragon.png
```

### Python Client (local VM)

```bash
python3 client.py --prompt "a small friendly dragon reading a bedtime story to forest animals, soft watercolor textures, pastel palette, rounded shapes, whimsical illustration, light paper grain, heartwarming mood" --output "dragon.png"
```

### Python Client (remote from your laptop)

Install dependency:
```bash
pip install requests
```

Run with public IP:
```bash
python3 client.py --host <Public-IP> --prompt "a small friendly dragon reading a bedtime story to forest animals, soft watercolor textures, pastel palette, rounded shapes, whimsical illustration, light paper grain, heartwarming mood" --output "dragon.png"
```

---

## 7. Test WebUI

After starting the WebUI container:

```bash
docker run --gpus all -p 7860:7860 sd3 python -m app.webui
```

Open in browser:
```
http://<Public-IP>:7860
```

You will see an interactive interface to enter prompts and view generated images.

---

## 8. Done

You now have a **Stable Diffusion 3.5 Turbo inference server** running in two modes:  

- **FastAPI REST API** (port 8000)  
- **Gradio WebUI** (port 7860)  

Both are GPU-accelerated inside Docker on Nebius Compute.
