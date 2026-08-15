# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

# Keep Python and Streamlit predictable inside the container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Run the web application as an unprivileged user.
RUN groupadd --system appgroup \
    && useradd --system --gid appgroup --create-home appuser

# Install dependencies before copying the source so Docker can cache this layer.
COPY requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appgroup . .

# Logging needs a writable directory. The model directory can either be copied
# into the image or mounted read-only when the container is started.
RUN mkdir -p /app/logs /app/models \
    && chown -R appuser:appgroup /app/logs /app/models

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", "--browser.gatherUsageStats=false"]
