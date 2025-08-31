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
