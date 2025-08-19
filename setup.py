"""Setup script for Wyoming Whisper TensorRT."""

from pathlib import Path
from setuptools import find_packages, setup

this_dir = Path(__file__).parent
module_dir = this_dir / "wyoming_whisper_trt"

requirements = []
requirements_path = this_dir / "requirements.txt"
if requirements_path.is_file():
    with open(requirements_path, "r", encoding="utf-8") as requirements_file:
        requirements = requirements_file.read().splitlines()

version_path = module_dir / "VERSION"
if version_path.is_file():
    with open(version_path, "r", encoding="utf-8") as version_file:
        version = version_file.read().strip()
else:
    version = "1.0.0"

setup(
    name="wyoming-whisper-trt",
    version=version,
    description="Wyoming protocol server for Whisper TensorRT",
    url="https://github.com/dwu/whisper-tensorrt",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="wyoming whisper tensorrt speech recognition",
    entry_points={
        "console_scripts": [
            "wyoming-whisper-trt = wyoming_whisper_trt.__main__:main",
        ]
    },
)