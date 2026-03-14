# Simple Transcription Pipeline

A minimal speech-to-text pipeline built with:
- FastAPI
- faster-whisper
- ffmpeg

## What it does

1. Accepts an uploaded audio file
2. Converts it to mono 16kHz WAV
3. Transcribes it with faster-whisper
4. Returns structured JSON with:
   - full text
   - timestamps
   - metadata
5. Saves a copy of the result in `outputs/`

## Requirements

- Python 3.10+
- ffmpeg installed and available in PATH

## Install

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt