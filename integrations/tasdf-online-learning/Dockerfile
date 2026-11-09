FROM python:3.10-slim

# CPU wheel by default; for GPU build with
#   --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_VERSION=2.7.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9090 \
    STATE_DIR=/data/tasdf-online

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL} \
    && pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser \
    && mkdir -p /data/tasdf-online && chown appuser /data/tasdf-online

COPY tasdf ./tasdf
COPY tasdf_online ./tasdf_online
USER appuser

EXPOSE 9090
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '9090') + '/health', timeout=2)" || exit 1

# exactly one worker: the model, the training queue and the registry live in this process
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads ${GUNICORN_THREADS:-4} --timeout ${GUNICORN_TIMEOUT:-300} --access-logfile - --error-logfile - 'tasdf_online.app:create_app()'"]
