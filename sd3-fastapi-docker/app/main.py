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
