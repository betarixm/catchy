FROM ghcr.io/betarixm/catchy-ubuntu:24.04

# Codex
RUN npm i -g @openai/codex

ENV CODEX_HOME=/metadata/.codex
COPY config.toml /metadata/.codex/config.toml
WORKDIR /workspace
