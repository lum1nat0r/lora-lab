FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[score]"

# Face models: YuNet for detection, ArcFace (buffalo_l) for embeddings.
# Downloaded at build time so the container works without network access.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && mkdir -p /opt/face \
 && curl -fsSL -o /opt/face/yunet.onnx \
      https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
 && echo "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  /opt/face/yunet.onnx" | sha256sum -c - \
 && curl -fsSL -o /tmp/buffalo_l.zip \
      https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
 && python -c "import shutil,zipfile; z=zipfile.ZipFile('/tmp/buffalo_l.zip'); n=next(n for n in z.namelist() if n.endswith('w600k_r50.onnx')); z.extract(n, '/tmp/face'); shutil.move('/tmp/face/' + n, '/opt/face/w600k_r50.onnx')" \
 && test -s /opt/face/w600k_r50.onnx \
 && rm -rf /tmp/face \
 && rm -f /tmp/buffalo_l.zip \
 && apt-get purge -y curl \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*

COPY config ./defaults
COPY workflows ./workflows
COPY docker/entrypoint.sh /usr/local/bin/lora-lab-entrypoint
RUN chmod +x /usr/local/bin/lora-lab-entrypoint

ENV LORA_LAB_DEFAULTS=/app/defaults \
    ARCFACE_MODEL=/opt/face/w600k_r50.onnx \
    YUNET_MODEL=/opt/face/yunet.onnx
EXPOSE 8099

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8099/healthz')"

ENTRYPOINT ["/usr/local/bin/lora-lab-entrypoint"]
CMD ["lora-lab", "--root", "/app", "serve", "--host", "0.0.0.0", "--port", "8099"]
