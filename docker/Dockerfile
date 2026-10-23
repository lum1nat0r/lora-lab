FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY config ./defaults
COPY workflows ./workflows
COPY docker/entrypoint.sh /usr/local/bin/lora-lab-entrypoint
RUN chmod +x /usr/local/bin/lora-lab-entrypoint

ENV LORA_LAB_DEFAULTS=/app/defaults
EXPOSE 8099

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8099/healthz')"

ENTRYPOINT ["/usr/local/bin/lora-lab-entrypoint"]
CMD ["lora-lab", "--root", "/app", "serve", "--host", "0.0.0.0", "--port", "8099"]
