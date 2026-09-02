# ---------- Base image ----------
FROM python:3.11-slim

# ---------- System dependencies ----------
RUN apt-get update && apt-get install -y \
    curl \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Install Node.js (LTS) ----------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm --version \
    && node --version

# ---- install uv + uvx ----
RUN pip install --no-cache-dir uv

# ==================================================
# ADD user (UID/GID must match host)
# ==================================================
ARG USERNAME=acradin
ARG USER_UID=8034
ARG USER_GID=1001

RUN groupadd -g ${USER_GID} ${USERNAME} || true \
    && useradd -m -u ${USER_UID} -g ${USER_GID} ${USERNAME}

# ---------- Set workdir ----------
WORKDIR /app

# ---------- Python dependencies ----------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Node dependencies ----------
COPY package.json package-lock.json ./
RUN npm ci --omit=dev

# ---------- Copy application code ----------
COPY . .

# ---------- Runtime directories & permissions ----------
RUN mkdir -p /workspace/agent /workspace/shared \
    && chown -R ${USERNAME}:${USER_GID} /app /workspace

# ---------- Environment defaults ----------
ENV PYTHONUNBUFFERED=1 \
    WORKSPACE=/workspace \
    WORKSPACE_AGENT=/workspace/agent \
    WORKSPACE_SHARED=/workspace/shared \
    PROMPT_DIR=/app/prompt \
    MCP_SERVER_CONFIG=/app/tool/mcp_servers/server_config.json \
    MCP_TOOL_BAN_LIST=/app/tool/mcp_servers/tool_ban_list.json \
    HOST=0.0.0.0 \
    PORT=8000 \
    PATH="/home/${USERNAME}/.local/bin:${PATH}" \
    AGENT_DB_PATH=/workspace/db/agent/agent.db \
    SHARED_DB_PATH=/workspace/db/shared/shared.db

# ---------- Switch to non-root user ----------
USER ${USERNAME}

# ---------- Expose port ----------
EXPOSE 8000

# ---------- Entrypoint ----------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
