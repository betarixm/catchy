FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# System packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    # Codex runtime
    bubblewrap nodejs npm \
    # Shell / everyday utilities
    bash-completion ca-certificates curl wget git jq less vim nano tmux \
    file xxd bsdextrautils zip unzip p7zip-full xz-utils zstd \
    procps psmisc htop \
    # Networking
    netcat-openbsd socat nmap dnsutils iproute2 iputils-ping \
    openssh-client tcpdump whois \
    # Binary analysis / reversing
    binutils elfutils patchelf gdb gdbserver ltrace strace \
    # Build tools
    build-essential clang make cmake ninja-build meson pkg-config \
    autoconf automake libtool \
    # Python base
    python3 python3-pip python3-dev python3-venv pipx \
    # Ruby / Java
    ruby ruby-dev default-jre-headless \
    # Forensics / steg / image analysis
    binwalk libimage-exiftool-perl steghide pngcheck imagemagick \
    # Filesystem forensics
    xfsprogs sleuthkit foremost dcfldd testdisk \
    # Audio / media / OCR
    ffmpeg sox tesseract-ocr tesseract-ocr-eng \
    # Crypto / cracking helpers
    openssl libssl-dev john \
    # Math libs required by flatter, cado-nfs, fpylll, gmpy2
    libgmp-dev libmpfr-dev libmpc-dev libfplll-dev libeigen3-dev \
    libopenblas-dev liblapack-dev \
    # z3 solver system lib
    libz3-dev \
    # CADO-NFS build deps
    libhwloc-dev libbz2-dev \
    # Source-build deps for stegseek
    libmhash-dev libmcrypt-dev libjpeg-dev zlib1g-dev \
    # Nested containers for challenge images
    podman buildah fuse-overlayfs slirp4netns uidmap \
    # Chromium runtime deps for Playwright / Chrome DevTools Protocol
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
    libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 libasound2t64 \
    libatspi2.0-0 fonts-liberation libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

# SageMath is useful for crypto / algebra challenges, but it is not always
# published for every Ubuntu release in the base image series.
RUN apt-get update \
    && if apt-cache policy sagemath | awk '/Candidate:/ { exit ($2 == "(none)") }'; then \
    apt-get install -y --no-install-recommends sagemath; \
    else \
    echo "sagemath is not available for this Ubuntu release; skipping"; \
    fi \
    && rm -rf /var/lib/apt/lists/*

# radare2
RUN git clone --depth=1 https://github.com/radareorg/radare2 /tmp/r2 \
    && cd /tmp/r2 \
    && bash sys/install.sh --install \
    && ldconfig \
    && r2 -v \
    && rm -rf /tmp/r2

# flatter (fast lattice reduction, Keegan Ryan's fork)
RUN git clone --depth=1 https://github.com/keeganryan/flatter /tmp/flatter \
    && cmake -S /tmp/flatter -B /tmp/flatter/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /tmp/flatter/build -j"$(nproc)" \
    && cmake --install /tmp/flatter/build \
    && ldconfig \
    && rm -rf /tmp/flatter

# uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh -o /tmp/install-uv.sh \
    && env UV_INSTALL_DIR=/usr/local/bin sh /tmp/install-uv.sh \
    && rm /tmp/install-uv.sh

# zsteg
RUN gem install zsteg

# stegseek (build from source; pre-built .deb is amd64 only)
RUN git clone --depth=1 https://github.com/RickdeJager/stegseek /tmp/stegseek \
    && cmake -S /tmp/stegseek -B /tmp/stegseek/build \
    && cmake --build /tmp/stegseek/build --target install -j"$(nproc)" \
    && rm -rf /tmp/stegseek

# Python CTF libraries
RUN uv python install 3.12 \
    && uv venv --seed --python 3.12 /opt/ctf-venv
ENV PATH="/opt/ctf-venv/bin:/usr/local/bin:$PATH"

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir \
    pwntools \
    pycryptodome \
    sympy \
    gmpy2 \
    requests \
    Pillow \
    z3-solver \
    pytesseract \
    scipy \
    numpy \
    angr \
    capstone \
    unicorn \
    ropgadget \
    tqdm \
    fpylll \
    flask \
    volatility3 \
    PyJWT \
    pyghidra \
    playwright \
    ipython

# Preinstall Playwright-managed Chromium for `playwright.chromium.launch()`
RUN mkdir -p "$PLAYWRIGHT_BROWSERS_PATH" \
    && chmod 755 "$PLAYWRIGHT_BROWSERS_PATH" \
    && python -m playwright install chromium

# Browser automation / Chrome DevTools Protocol
RUN chrome_url="$(curl -fsSL https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json \
    | jq -r '.channels.Stable.downloads.chrome[] | select(.platform == "linux64") | .url')" \
    && mkdir -p /opt/chrome-for-testing \
    && curl -fsSL "$chrome_url" -o /tmp/chrome-linux64.zip \
    && unzip -q /tmp/chrome-linux64.zip -d /opt/chrome-for-testing \
    && rm /tmp/chrome-linux64.zip \
    && chmod -R a+rX /opt/chrome-for-testing \
    && mv /opt/chrome-for-testing/chrome-linux64/chrome /opt/chrome-for-testing/chrome-linux64/chrome.real \
    && printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'exec /opt/chrome-for-testing/chrome-linux64/chrome.real \' \
    '  --disable-crash-reporter \' \
    '  --disable-crashpad \' \
    '  --noerrdialogs \' \
    '  --no-sandbox \' \
    '  --disable-dev-shm-usage \' \
    '  "$@"' \
    > /opt/chrome-for-testing/chrome-linux64/chrome \
    && chmod +x /opt/chrome-for-testing/chrome-linux64/chrome

RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'exec /opt/chrome-for-testing/chrome-linux64/chrome "$@"' \
    > /usr/local/bin/chrome \
    && chmod +x /usr/local/bin/chrome \
    && ln -sf /usr/local/bin/chrome /usr/local/bin/google-chrome \
    && ln -sf /usr/local/bin/chrome /usr/local/bin/google-chrome-stable \
    && printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    ': "${CHROME_REMOTE_DEBUGGING_PORT:=9222}"' \
    ': "${CHROME_USER_DATA_DIR:=/tmp/chrome-devtools-profile}"' \
    'exec chrome \' \
    '  --headless=new \' \
    '  --remote-debugging-address=0.0.0.0 \' \
    '  --remote-debugging-port="$CHROME_REMOTE_DEBUGGING_PORT" \' \
    '  --user-data-dir="$CHROME_USER_DATA_DIR" \' \
    '  --no-first-run \' \
    '  --no-default-browser-check \' \
    '  "$@"' \
    > /usr/local/bin/chrome-devtools \
    && chmod +x /usr/local/bin/chrome-devtools

# PyTorch (CPU-only) + Keras
RUN python -m pip install --no-cache-dir \
    torch \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    && python -m pip install --no-cache-dir keras

# RsaCtfTool (catch-all automated RSA/factoring attacks)
RUN git clone --depth=1 https://github.com/RsaCtfTool/RsaCtfTool /opt/RsaCtfTool \
    && python -m pip install --no-cache-dir /opt/RsaCtfTool

# CADO-NFS (Number Field Sieve: large integer factoring + DLP)
RUN git clone --depth=1 https://gitlab.inria.fr/cado-nfs/cado-nfs.git /opt/cado-nfs \
    && cd /opt/cado-nfs \
    && make -j"$(nproc)"

RUN printf '%s\n' \
    '#!/bin/bash' \
    'BUILD_PARENT=/opt/cado-nfs/build' \
    'H=$(hostname)' \
    'if [ ! -d "$BUILD_PARENT/$H" ]; then' \
    '  src=$(ls -d "$BUILD_PARENT"/* 2>/dev/null | head -1)' \
    '  [ -n "$src" ] && ln -sf "$src" "$BUILD_PARENT/$H"' \
    'fi' \
    'exec python3 /opt/cado-nfs/cado-nfs.py "$@"' \
    > /usr/local/bin/cado-nfs \
    && chmod +x /usr/local/bin/cado-nfs

# Support 32bit binary
RUN dpkg --add-architecture i386 && apt-get update && apt-get install --yes --no-install-recommends libc6:i386 libstdc++6:i386 libncurses6:i386 && rm -rf /var/lib/apt/lists/*
