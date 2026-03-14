

import json
import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from transcriber import AudioTranscriber

app = FastAPI(title="Simple Transcription Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp"
OUTPUT_DIR = "outputs"

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Use tiny or base for easier Render deployment
transcriber = AudioTranscriber(model_size="tiny", device="cpu", compute_type="int8")

@app.get("/")
def health():
    return {"status": "ok", "endpoints": ["/transcribe"]}

@app.post("/transcribe")
def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    allowed_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac", ".ogg"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    input_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = transcriber.process_file(input_path, temp_dir=UPLOAD_DIR)

        with open(os.path.join(OUTPUT_DIR, f"{result['job_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return JSONResponse(content=result)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        try:
            file.file.close()
        except Exception:
            pass