FROM python:3.13-slim

# Install system dependencies required by some python packages (like psycopg2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Set the working directory
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Sync dependencies using uv. This creates a virtualenv at /app/.venv
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY . .

# Add the uv virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Google Cloud Run provides the PORT environment variable (defaults to 8080)
ENV PORT=8080
EXPOSE $PORT

# Start the FastAPI server using the dynamically provided PORT
CMD uvicorn backend.src.api.server:app --host 0.0.0.0 --port $PORT
