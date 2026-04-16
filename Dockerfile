FROM python:3.12-slim AS base

# Install Rust toolchain for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Copy Rust crates first (for layer caching)
COPY rust/ rust/
COPY rust-text/ rust-text/

# Build Rust modules
RUN pip install maturin && \
    cd rust && maturin build --release && pip install target/wheels/*.whl && cd .. && \
    cd rust-text && maturin build --release && pip install target/wheels/*.whl && cd ..

# Copy Python package
COPY pyproject.toml MANIFEST.in ./
COPY sciscape/ sciscape/
COPY sos/ sos/

# Install Python package with web extras
RUN pip install ".[web,viz,arrow,openalex]"

# Default port for web UI
EXPOSE 8000

# Run web server
CMD ["uvicorn", "sciscape.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
