FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends socat && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
ENV HF_HOME=/app/.cache/huggingface PYTHONUNBUFFERED=1
# serve.py listens on 127.0.0.1:8765 inside the container; socat exposes it on 8766 so Docker can publish it
CMD ["sh", "-c", "socat TCP-LISTEN:8766,fork,reuseaddr TCP:127.0.0.1:8765 & exec python serve.py"]
