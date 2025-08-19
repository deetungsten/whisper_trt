"""Main entry point for Wyoming Whisper TensorRT server."""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from wyoming.info import AsrModel, AsrProgram, Attribution, Info
from wyoming.server import AsyncServer

from .handler import WhisperTRTEventHandler

_LOGGER = logging.getLogger(__name__)

async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="tiny.en",
        help="Name of Whisper model to use (default: tiny.en)",
    )
    parser.add_argument(
        "--model-dir",
        help="Directory to cache Whisper models (default: ~/.cache/whisper_trt)",
    )
    parser.add_argument(
        "--language", 
        help="Language for transcription (default: auto-detect)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )
    parser.add_argument(
        "--uri", 
        default="tcp://0.0.0.0:10300", 
        help="unix:// or tcp:// URI (default: tcp://0.0.0.0:10300)"
    )
    parser.add_argument(
        "--data-dir", 
        help="Data directory to check for downloaded models"
    )
    parser.add_argument(
        "--download-dir", 
        help="Directory to download models"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Log DEBUG messages"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    _LOGGER.debug("Args: %s", args)

    # Set model directory
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        model_dir = Path.home() / ".cache" / "whisper_trt"

    model_dir.mkdir(parents=True, exist_ok=True)

    # Create model info
    wyoming_info = Info(
        asr=[
            AsrProgram(
                name="whisper_trt",
                description="Whisper TensorRT speech-to-text",
                attribution=Attribution(
                    name="NVIDIA AI IOT", 
                    url="https://github.com/NVIDIA-AI-IOT/whisper_trt"
                ),
                installed=True,
                version="1.0.0",
                models=[
                    AsrModel(
                        name=args.model,
                        description=f"Whisper TensorRT {args.model} model",
                        attribution=Attribution(
                            name="OpenAI", 
                            url="https://github.com/openai/whisper"
                        ),
                        installed=True,
                        languages=["en"] if args.model.endswith(".en") else None,
                    )
                ],
            )
        ],
    )

    # Create event handler
    handler = WhisperTRTEventHandler(
        wyoming_info,
        args.model,
        model_dir=model_dir,
        language=args.language,
        beam_size=args.beam_size,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        download_dir=Path(args.download_dir) if args.download_dir else None,
    )

    # Start server
    async with AsyncServer.from_uri(args.uri) as server:
        _LOGGER.info("Ready")
        await server.run(handler.handle_event)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass