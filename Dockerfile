FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REPOLENS_DATA_DIR=/data \
    HF_HOME=/data/hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app \
    && mkdir /data && chown app /data

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu .

USER app
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "repolens.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
