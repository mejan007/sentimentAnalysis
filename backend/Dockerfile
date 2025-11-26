FROM python:3.12-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*


COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-install-project --no-dev

# Copy the source code into a 'backend' directory to preserve package structure
COPY . backend/

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]