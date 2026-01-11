# Use NVIDIA CUDA + PyTorch base image
FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python requirements
# Note: We install bitsandbytes for 4-bit/8-bit LoRA training
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir bitsandbytes accelerate peft dspy-ai

# Copy project files
COPY . .

# Ensure model and data directories exist
RUN mkdir -p model data reports

# Command to run the FastAPI server (Inference mode)
# Or swap to run main.py for a one-off retraining task
CMD ["uvicorn", "src.langchain_agent:app", "--host", "0.0.0.0", "--port", "8000"]