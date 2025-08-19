FROM ubuntu:20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    wyoming>=1.5.0 \
    torch \
    torchaudio \
    openai-whisper

# Create cache directory
RUN mkdir -p /root/.cache/whisper_trt

# Copy application code
COPY wyoming_whisper_trt/ ./wyoming_whisper_trt/
COPY setup.py .

# Install our package
RUN pip3 install -e .

# Create data directory and set permissions
RUN mkdir -p /data /root/.cache/whisper_trt && \
    chmod 755 /data /root/.cache/whisper_trt

# Expose port
EXPOSE 10300

# Set default command
ENTRYPOINT ["python3", "-m", "wyoming_whisper_trt"]
CMD ["--uri", "tcp://0.0.0.0:10300", "--model", "tiny.en"]