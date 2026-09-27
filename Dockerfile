# ============================================================================
# FOOTBALL CINEMATIC AI - Dockerfile for Render
# ----------------------------------------------------------------------------
# Solves the two production blockers:
#   1. FFmpeg / FFprobe are NOT available on Render's native Node runtime.
#      This image installs the full ffmpeg build + DejaVu fonts (needed by the
#      `drawtext` filter used for burned-in action text overlays).
#   2. Predictable, immutable runtime built from a lockfile, so deploys are
#      reproducible (no more "works locally, crashes on Render").
# ============================================================================

# ---------- Stage 1: Build the frontend (Vite) + bundle the server ----------
FROM node:22-bookworm-slim AS builder

WORKDIR /app

# Install ALL deps (dev included) needed for the build.
# --legacy-peer-deps is required: the project pins vite@^8 which conflicts with
# the esbuild peer range under strict npm resolution.
# npm ci uses package-lock.json when present for fully reproducible installs.
COPY package.json package-lock.json* bun.lock* ./
RUN if [ -f package-lock.json ]; then \
      npm ci --legacy-peer-deps --no-audit --no-fund; \
    else \
      npm install --legacy-peer-deps --no-audit --no-fund; \
    fi

# Copy the rest of the sources
COPY . .

# Monolithic deployment: the SPA and the API are served by the SAME Express
# server. The API base URL is pinned to the deployed Render backend so that every
# fetch call targets it (can be overridden at build time if needed).
ARG VITE_API_BASE_URL="https://footballcinematicai-v10.onrender.com"
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

# 1) Build the static frontend -> /app/dist
RUN npm run build

# 2) Pre-bundle server.ts -> /app/server.js (esbuild, externals kept).
#    This gives fast, deterministic cold starts on Render without tsx.
RUN npx esbuild server.ts --bundle --platform=node --target=node22 \
      --format=esm --packages=external --outfile=server.js

# ---------- Stage 2: Runtime ----------
FROM node:22-bookworm-slim AS runtime

# FFmpeg (with libx264, libfreetype for drawtext) + fonts + ffprobe + healthcheck.
# fontconfig + fonts-dejavu-core make `drawtext` work out of the box.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      fontconfig \
      fonts-dejavu-core \
      ca-certificates \
      curl \
      python3 \
      python3-pip \
      python3-venv \
      git \
      wget \
      unzip \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# Fail the build early if ffmpeg/ffprobe are missing
RUN ffmpeg -version && ffprobe -version && python3 --version

ENV NODE_ENV=production
WORKDIR /app

# --- Production dependencies only ---
COPY package.json package-lock.json* bun.lock* ./
RUN if [ -f package-lock.json ]; then \
      npm ci --omit=dev --legacy-peer-deps --no-audit --no-fund; \
    else \
      npm install --omit=dev --legacy-peer-deps --no-audit --no-fund; \
    fi

# --- App artifacts from the builder stage ---
COPY --from=builder /app/server.js ./server.js
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/public ./public
COPY yolo ./yolo
COPY models ./models

# Writable working + output directories.
#
# PERSISTENCE: /app/public/videos is the default media dir. On Render it is
# EPHEMERAL unless you attach a Persistent Disk. To persist, mount a disk at
# /var/data and set PUBLIC_DIR=/var/data/videos (see render.yaml). We pre-create
# BOTH locations so either configuration works without a code change.
# Intermediate FFmpeg segments stay in /tmp (disposable, never persisted).
RUN mkdir -p /tmp/football_engine/work public/videos /var/data/videos \
    && python3 -m venv /opt/yolo-venv \
    && /opt/yolo-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/yolo-venv/bin/pip install --no-cache-dir -r /app/yolo/requirements.txt \
    && /opt/yolo-venv/bin/python -c "import torch, torchvision; assert torch.__version__.startswith('2.6.0+cpu'), torch.__version__; assert torchvision.__version__.startswith('0.21.0+cpu'), torchvision.__version__; assert not torch.cuda.is_available(); print('CPU PyTorch OK:', torch.__version__, torchvision.__version__)" \
    && git clone --depth 1 https://github.com/megvii-research/ECCV2022-RIFE.git /opt/rife \
    && wget -q -O /tmp/rife.zip https://huggingface.co/aka7774/ECCV2022-RIFE/resolve/main/RIFE_trained_model_v3.6.zip \
    && mkdir -p /opt/rife/train_log \
    && unzip -q /tmp/rife.zip -d /opt/rife \
    && rm -f /tmp/rife.zip \
    && test -f /opt/rife/inference_video.py

# Storage defaults (override in the Render dashboard / render.yaml)
ENV STORAGE_DRIVER=local \
    PUBLIC_DIR=/app/public/videos \
    PYTHON_BIN=/opt/yolo-venv/bin/python \
    YOLO_MAX_SECONDS=120 \
    YOLO_STRIDE=3 \
    YOLO_TIMEOUT_MS=300000 \
    EVIDENCE_TIMEOUT_MS=120000 \
    RIFE_REPO=/opt/rife \
    RIFE_ENABLED=true \
    RIFE_FPS_MULTIPLIER=2

EXPOSE 3000

# Render injects $PORT at runtime; Express already reads process.env.PORT.
CMD ["node", "server.js"]
