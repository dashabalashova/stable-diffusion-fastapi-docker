# Stable Diffusion 3.5 Turbo – FastAPI + Docker Deployment Guide

This tutorial explains how to deploy **Stable Diffusion 3.5 Turbo** as a **FastAPI service** inside **Docker** on a GPU-enabled VM (Nebius Compute in this example).

The service has **two interfaces**:  
- **FastAPI** REST API – lightweight, script-friendly interface for programmatic access (cURL, Python client).  
- **Gradio WebUI** – interactive browser-based interface for quick prototyping and visualization.  

This way you can either call the model as a web service or interact with it in your browser.

⚙️ **Hardware requirements**

- A single modern NVIDIA GPU with at least **48 GB VRAM** is recommended (e.g., L40S).  
- You do **not** need the very latest GPUs (like H200) – Stable Diffusion 3.5 Turbo runs efficiently on mid/high-tier cards.  
- CPU-only execution is possible but extremely slow and not practical for real-time image generation.  

📦 **Model size**

The model [adamo1139/stable-diffusion-3.5-large-turbo-ungated](https://huggingface.co/adamo1139/stable-diffusion-3.5-large-turbo-ungated/tree/main/) is around **26 GB** when downloaded from Hugging Face Hub. Make sure you have enough disk space (at least 64 GB free recommended, considering model + cache + Docker layers).

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

The `Dockerfile` defines the container image and all the steps required to build the runtime environment:

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

* Base image: uses the official PyTorch image with CUDA 12.8 and cuDNN 9 for GPU acceleration.
* WORKDIR: sets /app as the working directory inside the container.
* Dependencies: installs requirements.txt plus the optimized xformers library for fast attention.
* ENV: enables hf_transfer for faster downloads from Hugging Face Hub.
* COPY app: copies your source code into the container.
* EXPOSE: opens ports 8000 (FastAPI) and 7860 (Gradio).
* CMD: by default starts the FastAPI server, but you can override the command at runtime (for example to launch Gradio WebUI instead).

### requirements.txt

The `requirements.txt` file lists all Python dependencies:

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

# Client
requests
```

* Hugging Face stack: model loading, tokenization, inference.
* TorchVision: essential computer vision utilities.
* API stack: FastAPI + Uvicorn to serve the model, Pillow to work with images, hf_transfer for faster downloads.
* Gradio: provides a simple web UI.
* Requests: required by the included Python client (client.py).

### app/sd3.py

The `app/sd3.py` file loads the Stable Diffusion 3.5 Turbo pipeline from Hugging Face and provides a helper function to generate images:

```python
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
```

* `MODEL_ID` points to the chosen model on Hugging Face Hub (adamo1139/stable-diffusion-3.5-large-turbo-ungated).
* Device selection: uses GPU (cuda) if available, otherwise CPU.
* `torch_dtype`: `float16` on GPU for efficiency, `float32` on CPU.
* `generate_image()` takes a text prompt, runs inference with a small number of steps (fast “turbo mode”), and returns a PIL image.
* `guidance_scale=0.0` disables classifier-free guidance for maximum speed.
* `generate_images()` generates one image per prompt from a list -- splits prompts into chunks (to fit GPU memory), processes each chunk in one batch call, returns a flat list of PIL images.

### app/main.py

The `app/main.py` file defines the FastAPI service:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from io import BytesIO
from .sd3 import generate_image, generate_images
import time
import zipfile, io
from pydantic import BaseModel

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

class PromptsRequest(BaseModel):
    prompts: list[str]
    chunk_size: int = 4

@app.post("/generate_images")
def generate_images_endpoint(req: PromptsRequest):
    images = generate_images(req.prompts, chunk_size=req.chunk_size)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i, img in enumerate(images):
            img_buf = io.BytesIO()
            img.save(img_buf, format="PNG")
            zf.writestr(f"image_{i}.png", img_buf.getvalue())
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=images.zip"}
    )
```

* `/` route provides a health check.
* `/generate` accepts a prompt, calls the model, and streams back a PNG image.
* `/generate_images` accepts a JSON body with a list of prompts and optional `chunk_size`, generates one image per prompt (batched in chunks to fit GPU), returns a zip archive with all images in PNG format.  

### app/webui.py

The `app/webui.py` file provides an interactive Gradio-based WebUI:

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

* Runs Gradio on port 7860.
* Lets you enter a text prompt and see generated images directly in the browser.

### Python client example

You can call the FastAPI server directly from Python. Replace `<HOST>` and `<PORT>` with your server address and port:

```python
import requests

api_url = "http://<HOST>:<PORT>/generate"
prompt = "a small friendly dragon reading a bedtime story to forest animals"

response = requests.get(api_url, params={"prompt": prompt})
if response.status_code == 200:
    with open("dragon.png", "wb") as f:
        f.write(response.content)
    print("✅ Saved as dragon.png")
else:
    print("❌ Error:", response.status_code, response.text)

```

This minimal client sends a text prompt to the `/generate` endpoint and saves the generated image as `dragon.png`. For a full CLI version with argument parsing, see `client.py`.

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

## 5. Run API (FastAPI)

Start the API server inside Docker (it will listen on port 8000):

```bash
docker run --gpus all -p 8000:8000 sd3 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Curl (from VM)

You can test the API directly from inside the VM using curl. This will send a GET request with your prompt and save the result as **dragon.png**:

```bash
curl "http://localhost:8000/generate?prompt=a+small+friendly+dragon+reading+a+bedtime+story+to+forest+animals,+soft+watercolor+textures,+pastel+palette,+rounded+shapes,+whimsical+illustration,+light+paper+grain,+heartwarming+mood" --output dragon.png
```

### Python Client (from VM)

Instead of curl, you can use the included Python client (client.py). Here spaces in the prompt can be written as normal text (no need for +):

```bash
python3 client.py --prompt "a small friendly dragon reading a bedtime story to forest animals, soft watercolor textures, pastel palette, rounded shapes, whimsical illustration, light paper grain, heartwarming mood" --output "dragon.png"
```

### Python Client (remote from your laptop)

If you want to call the API from your own laptop (not from inside the VM).

Install dependency:
```bash
pip install requests
```

Run the client with the public IP of your VM. This will connect to port 8000 on the server and save the output locally:
```bash
python3 client.py --host <Public-IP> --prompt "a small friendly dragon reading a bedtime story to forest animals, soft watercolor textures, pastel palette, rounded shapes, whimsical illustration, light paper grain, heartwarming mood" --output "dragon.png"
```

### Stop a running container

Show all running containers with names, IDs and exposed ports:
```bash
docker ps
```

Stop:
```bash
docker stop <container-name-or-id>
```

---

## 6. Run WebUI (Gradio)

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

## 7. Done

You now have a **Stable Diffusion 3.5 Turbo inference server** running in two modes:  

- **FastAPI REST API** (port 8000)  
- **Gradio WebUI** (port 7860)  

Both are GPU-accelerated inside Docker on Nebius Compute.

---

## 8. Benchmark results and analysis

We benchmarked two API modes on 20 prompts. Before running the benchmark scripts, ensure they are executable. From the repository root run:
```bash
chmod +x scripts/benchmark_generate.sh scripts/benchmark_chunks.sh
```

Then run benchmarks:
```bash
scripts/benchmark_generate.sh
scripts/benchmark_chunks.sh
```
- `/generate` -> one request per prompt, one image each  
- `/generate_images` -> one request with batched prompts (`chunk_size`)

### Results

| Mode / chunk_size | Time (s) | Throughput (img/sec) |
|-------------------|----------|-----------------------|
| `/generate`       | 45       | 0.44                  |
| chunk_size = 1    | 44       | 0.45                  |
| chunk_size = 2    | 45       | 0.44                  |
| chunk_size = 5    | 47       | 0.42                  |

### Analysis

- Throughput is nearly identical across all modes.  
- Encoder + pipeline overhead is small, so batching them saves little time.  
- API overhead (HTTP, JSON, PNG encoding, zip) is constant and masks small efficiency gains.
- Batching with chunk_size > 1 is valuable for convenience (sending multiple prompts in one request), but in current settings it does not significantly improve throughput.  

---

## 9. Clean up (optional)

Open [Nebius Console](https://console.nebius.com/compute). Find the VM you want to delete. Click the three vertical dots (⋮) on the instance row and choose **Delete** — in the confirmation dialog, check **Delete boot disk** if you also want the instance disk removed.
