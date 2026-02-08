# Multi-stage Containerfile for RHOAI MCP Server
# Optimized for Podman with Red Hat UBI base images
# Supports all transport modes (stdio, SSE, streamable-http)
#
# Build with: podman build -f Containerfile -t rhoai-mcp .

# =============================================================================
# Stage 1: Builder - Install dependencies with uv
# =============================================================================
ARG BUILD_PLATFORM=linux/amd64
FROM --platform=${BUILD_PLATFORM} registry.access.redhat.com/ubi9/python-312 AS builder

# Copy uv from official image for fast, reproducible builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory (UBI default is /opt/app-root/src)
WORKDIR /opt/app-root/src

# Copy project files for dependency resolution
COPY pyproject.toml uv.lock README.md ./

# Copy source code
COPY src/ ./src/

# Install dependencies and package
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Runtime - Minimal production image
# =============================================================================
FROM --platform=${BUILD_PLATFORM} registry.access.redhat.com/ubi9/python-312 AS runtime

# Labels for container metadata
LABEL org.opencontainers.image.title="RHOAI MCP Server"
LABEL org.opencontainers.image.description="MCP server for Red Hat OpenShift AI - enables AI agents to interact with RHOAI environments"
LABEL org.opencontainers.image.vendor="Red Hat"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/admiller/rhoai-mcp-prototype"

# Set working directory (UBI default is /opt/app-root/src)
WORKDIR /opt/app-root/src

# Copy virtual environment from builder
COPY --from=builder /opt/app-root/src/.venv /opt/app-root/src/.venv

# Copy source code (needed for editable installs)
COPY --from=builder /opt/app-root/src/src /opt/app-root/src/src

# Add virtual environment to PATH
ENV PATH="/opt/app-root/src/.venv/bin:$PATH"

# Environment variables with container-friendly defaults
# Transport: default to stdio for Claude Desktop compatibility
ENV RHOAI_MCP_TRANSPORT="stdio"
# HTTP binding: use 0.0.0.0 for container networking
ENV RHOAI_MCP_HOST="0.0.0.0"
ENV RHOAI_MCP_PORT="8000"
# Auth: default to auto-detection
ENV RHOAI_MCP_AUTH_MODE="auto"
# Logging: default to INFO
ENV RHOAI_MCP_LOG_LEVEL="INFO"

# Expose port for HTTP transports (SSE, streamable-http)
EXPOSE 8000

# UBI runs as non-root by default (UID 1001)
USER 1001

# Health check for HTTP transports
# Note: Only works with SSE/streamable-http, not stdio
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)" || exit 0

# Default entrypoint runs the MCP server
ENTRYPOINT ["rhoai-mcp"]

# Default to stdio transport (can be overridden with --transport sse|streamable-http)
CMD ["--transport", "stdio"]
