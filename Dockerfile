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
