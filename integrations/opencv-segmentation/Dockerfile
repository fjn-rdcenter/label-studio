FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=9090

WORKDIR /app

# Trust the corporate proxy CA (DER or PEM) so pip can reach PyPI.
COPY --from=certs FJS_SSL_Certificates.crt /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt
RUN if ! grep -q -- '-----BEGIN CERTIFICATE-----' /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt; then \
      openssl x509 -inform DER -in /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt -outform PEM -out /tmp/fjs.pem \
      && mv /tmp/fjs.pem /usr/local/share/ca-certificates/FJS_SSL_Certificates.crt; \
    fi \
    && update-ca-certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 appuser

COPY opencv_segmentation ./opencv_segmentation
USER appuser

EXPOSE 9090
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '9090') + '/health', timeout=2)" || exit 1

CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY:-2} --threads ${GUNICORN_THREADS:-2} --timeout ${GUNICORN_TIMEOUT:-60} --access-logfile - --error-logfile - opencv_segmentation.app:app"]
