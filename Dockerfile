FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade NumPy first (L4T container has old version)
RUN pip3 install --no-cache-dir --upgrade numpy>=1.22.0

# Install dependencies step by step for better error handling
RUN pip3 install --no-cache-dir wyoming>=1.5.0

# Install whisper dependencies first
RUN pip3 install --no-cache-dir openai-whisper

# Install whisper_trt from source
RUN git clone https://github.com/NVIDIA-AI-IOT/whisper_trt.git /tmp/whisper_trt && \
    cd /tmp/whisper_trt && \
    pip3 install --no-cache-dir -e . && \
    rm -rf /tmp/whisper_trt

# Copy application code
COPY wyoming_whisper_trt/ ./wyoming_whisper_trt/
COPY setup.py .

# Install the package
RUN pip3 install -e .

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 10300

# Set default command
ENTRYPOINT ["python3", "-m", "wyoming_whisper_trt"]
CMD ["--uri", "tcp://0.0.0.0:10300", "--model", "tiny.en"]