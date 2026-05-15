FROM ghcr.io/betarixm/catchy-ubuntu:24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /workspace /metadata \
    && chmod 1777 /workspace /metadata

# Claude Code
WORKDIR /tmp
RUN curl -fsSL https://claude.ai/install.sh -o /tmp/install-claude-code.sh \
    && bash /tmp/install-claude-code.sh \
    && rm /tmp/install-claude-code.sh \
    && chmod 755 /root

ENV CLAUDE_CONFIG_DIR=/metadata/.claude
ENV PATH="/root/.local/bin:${PATH}"
COPY settings.json /metadata/.claude/settings.json
WORKDIR /workspace
