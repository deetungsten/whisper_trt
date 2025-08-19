# Wyoming Whisper TensorRT

A [Wyoming protocol](https://github.com/rhasspy/wyoming) server for speech recognition using [Whisper TensorRT](https://github.com/NVIDIA-AI-IOT/whisper_trt), optimized for NVIDIA Jetson devices and GPUs.

## Features

- High-performance speech recognition using Whisper TensorRT
- ~3x faster inference compared to standard Whisper
- ~60% memory reduction
- Compatible with Home Assistant via Wyoming protocol
- Support for multiple Whisper models
- Automatic TensorRT engine caching

## Requirements

### NVIDIA Jetson Orin Nano (Recommended)

This integration is optimized for NVIDIA Jetson devices, especially the Orin Nano:
- JetPack 5.1+ (Ubuntu 20.04)
- Python 3.8+
- PyTorch (pre-installed in L4T containers)
- TensorRT (pre-installed in L4T containers)

### Other NVIDIA GPUs

- NVIDIA GPU with CUDA support
- Python 3.8+
- PyTorch with CUDA support
- TensorRT

## Installation

### Local Installation (Jetson Orin Nano)

1. Clone this repository:
   ```bash
   git clone https://github.com/deetungsten/whisper_trt
   cd whisper_trt
   ```

2. Run the setup script:
   ```bash
   script/setup
   ```

3. Activate the virtual environment:
   ```bash
   source .venv/bin/activate
   ```

### Alternative Installation (Manual)

For Jetson devices, you may prefer installing dependencies manually:
```bash
# Install Wyoming
pip3 install wyoming

# Install Whisper TensorRT
git clone https://github.com/NVIDIA-AI-IOT/whisper_trt.git
cd whisper_trt
pip3 install -e .
```

### Docker Installation

The Docker image now includes native whisper_trt support built from source:

```bash
docker-compose up -d
```

This will:
- Build the L4T container with whisper_trt included
- Enable GPU access for TensorRT acceleration
- Cache models in `~/.cache/whisper_trt`

**Note**: The first model load will build the TensorRT engine which takes a few minutes. Subsequent runs will use the cached engine for fast startup.

## Usage

### Command Line

Start the server with default settings:
```bash
script/run --model tiny.en --language en
```

Available options:
- `--model`: Whisper model to use (tiny.en, base.en, small.en, medium.en, large-v2, etc.)
- `--language`: Language for transcription (default: auto-detect)
- `--uri`: Server URI (default: tcp://0.0.0.0:10300)
- `--beam-size`: Beam size for decoding (default: 5)
- `--model-dir`: Directory to cache models (default: ~/.cache/whisper_trt)
- `--debug`: Enable debug logging

### Docker

```bash
docker run -it --rm \
  --runtime=nvidia \
  -p 10300:10300 \
  -v ~/.cache/whisper_trt:/root/.cache/whisper_trt \
  wyoming-whisper-trt \
  --model tiny.en --language en
```

## Home Assistant Integration

1. Add the following to your Home Assistant `configuration.yaml`:

```yaml
stt:
  - platform: wyoming
    uri: tcp://YOUR_SERVER_IP:10300
```

2. Restart Home Assistant

3. Configure voice assistants to use the new STT provider

## Models

Supported Whisper models:
- `tiny.en` - Fastest, English only
- `base.en` - Good balance, English only  
- `small.en` - Better accuracy, English only
- `medium.en` - High accuracy, English only
- `tiny` - Fastest, multilingual
- `base` - Good balance, multilingual
- `small` - Better accuracy, multilingual
- `medium` - High accuracy, multilingual
- `large-v2` - Best accuracy, multilingual

The first time you use a model, TensorRT will build an optimized engine which may take a few minutes. Subsequent runs will use the cached engine for fast startup.

## Performance

Typical performance on NVIDIA Jetson Orin Nano:
- `tiny.en`: ~0.64 seconds
- `base.en`: ~0.86 seconds

Results will vary based on audio length and hardware.

## Troubleshooting

### CUDA/TensorRT Issues

Ensure you have:
- NVIDIA drivers installed
- CUDA toolkit
- TensorRT libraries
- PyTorch with CUDA support

### Model Loading Issues

- Check available disk space for model caching
- Verify network connectivity for initial model downloads
- Check permissions on cache directory

### Performance Issues

- Ensure GPU is being utilized (check `nvidia-smi`)
- Monitor memory usage
- Consider using smaller models if memory is limited

## Development

To set up for development:

1. Clone the repository
2. Run `script/setup`
3. Make your changes
4. Test with `script/run --debug`

## License

MIT License - see LICENSE file for details.

## Credits

- [Whisper TensorRT](https://github.com/NVIDIA-AI-IOT/whisper_trt) by NVIDIA AI IOT
- [Wyoming](https://github.com/rhasspy/wyoming) by Rhasspy
- [OpenAI Whisper](https://github.com/openai/whisper) by OpenAI