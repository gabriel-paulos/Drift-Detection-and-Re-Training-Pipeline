# --------------------------------------------------------------------------
# STAGE 1: Builder - Installs uv and all dependencies for fast, cached builds
# --------------------------------------------------------------------------
FROM python:3.10-slim AS builder

# Install uv (This is generally faster than 'pip install uv' in a clean image)
# We copy the binary from Astral's pre-built image for maximum efficiency.
COPY --from=ghcr.io/astral-sh/uv:latest /usr/local/bin/uv /usr/local/bin/

# Set working directory
WORKDIR /app

# Copy dependency file(s) first to leverage Docker's layer caching.
# Note: For uv's full power, you should ideally use pyproject.toml and uv.lock.
# Since we only have requirements.txt currently, we'll use uv's pip compatibility.
COPY requirements.txt .

# Use uv's pip interface to install dependencies from requirements.txt
# --system installs into the default Python site-packages, avoiding a venv for simplicity.
# --no-cache-dir saves space during the build.
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# --------------------------------------------------------------------------
# STAGE 2: Runtime - A lean image for production serving
# --------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the application code and the installed packages (site-packages) from the builder stage
# /usr/local/lib/python3.10/site-packages is where '--system' installs packages
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /app /app

# The necessary utilities (like uvicorn) are now available on the PATH
# because they were installed into the site-packages in the builder stage.

# Expose the port for the FastAPI inference server
EXPOSE 8000

# Command to run the inference server
CMD ["uvicorn", "model.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]