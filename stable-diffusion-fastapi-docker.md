# Stable Diffusion 3.5 Turbo – FastAPI + Docker Deployment Guide

This tutorial explains how to deploy **Stable Diffusion 3.5 Turbo** as a **FastAPI service** inside **Docker** on a GPU-enabled VM (Nebius Compute in this example).

![Dragon](images/dragon.png)

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

Create a working directory:

```bash
mkdir sd3-fastapi-docker
cd sd3-fastapi-docker
```

---

## 3. Dockerfile

Create `Dockerfile` that defines the container image (you can do this with `nano Dockerfile` and copy-paste the code below):

```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install -U xformers --index-url https://download.pytorch.org/whl/cu128

COPY app ./app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

```

We use the official PyTorch base image with CUDA 12.8 and cuDNN 9 runtime to enable GPU acceleration. The requirements.txt file is copied and installed, followed by xformers (optimized attention kernels for diffusion models). Next, we copy the application source code (app/ folder with main.py and sd3.py) into the container so that FastAPI can be launched inside the image. Finally, port 8000 is exposed, and uvicorn is set as the container entrypoint. Finally, the application code is added, port 8000 is exposed, and uvicorn launches the FastAPI server as the container entrypoint.

---

## 4. Requirements

Create `requirements.txt` that lists all the Python dependencies for the project:

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
```

* The Hugging Face stack provides the core functionality for Stable Diffusion (model loading, tokenization, inference).
* TorchVision adds essential computer vision utilities.
* The API stack (FastAPI + Uvicorn) enables serving the model as a web service, while Pillow handles image saving/streaming.
* hf_transfer speeds up downloading large models from Hugging Face.

---

## 5. FastAPI Application

Create app folder:

```bash
mkdir app
cd app
```

Create `sd3.py`:

```python
import torch
from diffusers import StableDiffusion3Pipeline

MODEL_ID = "adamo1139/stable-diffusion-3.5-large-turbo-ungated"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model: {MODEL_ID} ...")
pipe = StableDiffusion3Pipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None,        # disable safety checker
).to(device)

def generate_image(prompt: str, num_inference_steps: int = 4):
    image = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=0.0       # turbo mode without CFG
    ).images[0]
    return image
```

This module loads the Stable Diffusion 3.5 Turbo pipeline from Hugging Face and provides a helper function to generate images:

* MODEL_ID points to the chosen model on Hugging Face (adamo1139/stable-diffusion-3.5-large-turbo-ungated).
* The pipeline is loaded once at startup and moved to GPU (cuda) if available.
* torch_dtype=torch.float16 ensures efficient GPU usage.
* safety_checker=None disables filtering of generated images (for maximum performance).
* The function generate_image(prompt, num_inference_steps) takes a text prompt and returns the generated PIL image.
* guidance_scale=0.0 is used for turbo mode (no classifier-free guidance).

Create `main.py`:

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

This module defines the FastAPI service:
* The / route provides a simple health check.
* The /generate route accepts a text prompt (?prompt=...) and calls generate_image().
* The generated image is stored in an in-memory buffer (BytesIO) and streamed back as PNG.

---

## 6. Build and Run Docker Container

Go back to project root:

```bash
cd ..
```

Enable Docker for User: 

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

Build and Run Docker Container:

```bash
docker build -t sd3-fastapi .
docker run --gpus all -p 8000:8000 sd3-fastapi
```

---

## 7. Test the API

### From VM itself:

```bash
curl "http://localhost:8000/generate?prompt=a+small+friendly+dragon+reading+a+bedtime+story+to+forest+animals,+soft+watercolor+textures,+pastel+palette,+rounded+shapes,+whimsical+illustration,+light+paper+grain,+heartwarming+mood" --output dragon.png
```

### From local machine:

```bash
curl "http://<Public-IP>:8000/generate?prompt=a+small+friendly+dragon+reading+a+bedtime+story+to+forest+animals,+soft+watercolor+textures,+pastel+palette,+rounded+shapes,+whimsical+illustration,+light+paper+grain,+heartwarming+mood" --output dragon.png
```

---

## 8. Done

You now have a **Stable Diffusion 3.5 Turbo FastAPI inference server** running inside **Docker** with GPU acceleration.  

Access your service at:

```
http://<Public-IP>:8000/generate?prompt=your+prompt+here
```
