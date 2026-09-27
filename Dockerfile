FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    GRADIO_ANALYTICS_ENABLED=False \
    PRELOAD_MODELS=1

WORKDIR /srv

# Dependencies first so code changes do not reinstall torch
COPY app/requirements.txt app/requirements.txt
RUN pip install gradio==6.28.0 -r app/requirements.txt

COPY src/ src/
COPY app/ app/

RUN useradd --create-home appuser && mkdir -p /cache/huggingface && chown -R appuser /cache
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/', timeout=4)"

CMD ["python", "-m", "app.app"]
