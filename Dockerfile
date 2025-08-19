FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:$CUDA_HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64
# Force rebuild - updated error handling for torch2trt/whisper_trt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    wget \
    build-essential \
    cmake \
    libopenblas-dev \
    libblas3 \
    libblas-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip3 install --upgrade pip setuptools wheel

# Install basic dependencies first
RUN pip3 install --no-cache-dir wyoming>=1.5.0

# Install specific numpy version compatible with L4T
RUN pip3 install --no-cache-dir "numpy>=1.19.4,<1.25.0"

# Install whisper dependencies 
RUN pip3 install --no-cache-dir \
    numba \
    librosa \
    more-itertools \
    transformers>=4.19.0 \
    ffmpeg-python==0.2.0 \
    tiktoken \
    openai-whisper

# Try to install torch2trt and whisper_trt, but don't fail if TensorRT issues occur
RUN set +e; \
    git clone https://github.com/NVIDIA-AI-IOT/torch2trt /tmp/torch2trt && \
    cd /tmp/torch2trt && \
    python3 setup.py install --user; \
    if [ $? -ne 0 ]; then \
        echo "torch2trt installation failed, will use runtime mounting"; \
    fi; \
    cd / && rm -rf /tmp/torch2trt; \
    set -e

RUN set +e; \
    git clone https://github.com/NVIDIA-AI-IOT/whisper_trt.git /tmp/whisper_trt && \
    cd /tmp/whisper_trt && \
    python3 setup.py install --user; \
    if [ $? -ne 0 ]; then \
        echo "whisper_trt installation failed, will use runtime mounting"; \
    fi; \
    cd / && rm -rf /tmp/whisper_trt; \
    set -e

# Create fallback directories for runtime mounting
RUN mkdir -p /opt/whisper_trt /opt/torch2trt

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