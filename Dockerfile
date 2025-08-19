FROM python:3.9-slim

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install basic dependencies
RUN pip3 install --no-cache-dir \
    wyoming>=1.5.0 \
    torch \
    torchaudio \
    openai-whisper

# Copy application code
COPY wyoming_whisper_trt/ ./wyoming_whisper_trt/
COPY setup.py .

# Create a stub whisper_trt module for non-Jetson environments
RUN mkdir -p /usr/local/lib/python3.9/site-packages/whisper_trt && \
    echo "# Stub whisper_trt for non-Jetson environments" > /usr/local/lib/python3.9/site-packages/whisper_trt/__init__.py && \
    echo "def load_trt_model(model_name): raise ImportError('whisper_trt requires NVIDIA Jetson hardware')" >> /usr/local/lib/python3.9/site-packages/whisper_trt/__init__.py

# Install the package
RUN pip3 install -e .

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 10300

# Set default command
ENTRYPOINT ["python3", "-m", "wyoming_whisper_trt"]
CMD ["--uri", "tcp://0.0.0.0:10300", "--model", "tiny.en"]