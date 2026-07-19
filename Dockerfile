# linux/amd64 Python 3.12.13 slim-bookworm pin. Update the digest deliberately.
FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    UV_PROJECT_ENVIRONMENT=/opt/qdrant-mcp \
    UV_CACHE_DIR=/tmp/uv-cache

WORKDIR /build

RUN mkdir /tmp/uv \
    && python -m pip download --no-cache-dir --no-deps --only-binary=:all: \
      --dest /tmp/uv uv==0.11.29 \
    && echo "eec03a8b63d55915694db3af4e91324b39ced49e2aeac7af37851c7eb3f470ea  /tmp/uv/uv-0.11.29-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl" \
      | sha256sum --check --strict \
    && python -m pip install --no-cache-dir --no-deps /tmp/uv/*.whl \
    && rm -rf /tmp/uv

COPY scripts/install_fastembed_model.py ./scripts/install_fastembed_model.py

RUN python scripts/install_fastembed_model.py --destination /opt/fastembed-model

COPY pyproject.toml uv.lock README.md LICENSE NOTICE ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable \
    && rm -rf /tmp/*


FROM python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

ARG BUILD_SHA=development
ARG SOURCE_FINGERPRINT=development
ARG IMAGE_VERSION=2.0.0-dev

LABEL org.opencontainers.image.title="MADPANDA3D Qdrant MCP" \
      org.opencontainers.image.description="Authenticated dual-mode Qdrant Model Context Protocol server" \
      org.opencontainers.image.source="https://github.com/MADPANDA3D/QDRANT-MCP" \
      org.opencontainers.image.revision="${BUILD_SHA}" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.licenses="Apache-2.0" \
      com.madpanda.source-fingerprint="${SOURCE_FINGERPRINT}"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
      antiword \
      ca-certificates \
      poppler-utils \
      tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
    && test "$(id -u app)" = "10001" \
    && test "$(id -g app)" = "10001"

WORKDIR /app

COPY --from=builder --chown=10001:10001 /opt/qdrant-mcp /opt/qdrant-mcp
COPY --from=builder --chown=10001:10001 /opt/fastembed-model \
  /opt/qdrant-mcp/models/all-MiniLM-L6-v2
COPY --chown=10001:10001 scripts/runtime_smoke.py /app/scripts/runtime_smoke.py
COPY LICENSE NOTICE /usr/share/licenses/mad-mcp-qdrant/
RUN chmod 0444 /usr/share/licenses/mad-mcp-qdrant/LICENSE \
    /usr/share/licenses/mad-mcp-qdrant/NOTICE \
    /opt/qdrant-mcp/models/all-MiniLM-L6-v2/* \
    && chmod 0555 /opt/qdrant-mcp/models/all-MiniLM-L6-v2

ENV PATH="/opt/qdrant-mcp/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    FASTEMBED_MODEL_PATH=/opt/qdrant-mcp/models/all-MiniLM-L6-v2 \
    FASTEMBED_MODEL_REVISION=qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079 \
    FASTMCP_SERVER_HOST=0.0.0.0 \
    FASTMCP_SERVER_PORT=8000 \
    MCP_MODE=standalone \
    MCP_SERVER_VERSION=2.0.0 \
    MCP_BUILD_SHA="${BUILD_SHA}" \
    MCP_SOURCE_FINGERPRINT="${SOURCE_FINGERPRINT}" \
    MCP_IMAGE_REFERENCE=development \
    MCP_IMAGE_DIGEST=""

EXPOSE 8000
USER 10001:10001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import json,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); r=o.open('http://127.0.0.1:8000/health',timeout=3); b=r.read(65537); r.close(); p=json.loads(b); raise SystemExit(0 if len(b)<=65536 and p.get('ok') is True and p.get('version')=='2.0.0' and isinstance(p.get('build_sha'),str) and int(p.get('tool_count',0))>0 else 1)"]

CMD ["mad-mcp-qdrant", "--transport", "streamable-http"]
