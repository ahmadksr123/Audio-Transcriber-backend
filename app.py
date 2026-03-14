

# import json
# import os
# import shutil
# from pathlib import Path

# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware

# from schemas import TranscriptResponse, JobSubmittedResponse, JobStatusResponse
# from transcriber import AudioTranscriber, job_store, JobStatus

# APP_NAME = "Simple Transcription Pipeline"
# UPLOAD_DIR = "temp"
# OUTPUT_DIR = "outputs"

# Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
# Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# app = FastAPI(title=APP_NAME)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://127.0.0.1:5500",
#         "http://localhost:5500",
#         "http://127.0.0.1:8001",
#         "http://localhost:8001",
#         "http://localhost:3000",
#     ],
#     allow_credentials=False,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# transcriber = AudioTranscriber(model_size="small", device="cpu", compute_type="int8")

# ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".mp4", ".aac", ".flac", ".ogg"}


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# def _validate_extension(filename: str) -> str:
#     """Raise 400 if the extension is not supported; otherwise return the ext."""
#     ext = os.path.splitext(filename)[1].lower()
#     if ext not in ALLOWED_EXTENSIONS:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported file type: '{ext}'. "
#                    f"Allowed: {sorted(ALLOWED_EXTENSIONS)}",
#         )
#     return ext


# def _save_upload(file: UploadFile, dest: str) -> None:
#     """Stream an uploaded file to *dest* on disk."""
#     with open(dest, "wb") as buf:
#         shutil.copyfileobj(file.file, buf)


# def _persist_result(result: dict) -> None:
#     """Write a completed transcription result to the outputs directory."""
#     out_path = os.path.join(OUTPUT_DIR, f"{result['job_id']}.json")
#     with open(out_path, "w", encoding="utf-8") as f:
#         json.dump(result, f, ensure_ascii=False, indent=2)


# # ---------------------------------------------------------------------------
# # Routes
# # ---------------------------------------------------------------------------

# @app.get("/")
# def health_check():
#     return {
#         "message": APP_NAME,
#         "status": "ok",
#         "endpoints": {
#             "POST /transcribe":       "Synchronous transcription (blocks until done)",
#             "POST /transcribe/async": "Submit a background job; returns job_id immediately",
#             "GET  /jobs/{job_id}":    "Poll background job status / result",
#             "GET  /jobs":             "List all known jobs",
#         },
#     }


# # -- Synchronous endpoint (unchanged behaviour) ----------------------------

# @app.post("/transcribe", response_model=TranscriptResponse)
# def transcribe_audio(file: UploadFile = File(...)):
#     """
#     Upload an audio file and receive the full transcript synchronously.
#     Suitable for short clips; for long recordings use POST /transcribe/async.
#     """
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     _validate_extension(file.filename)
#     temp_input_path = os.path.join(UPLOAD_DIR, file.filename)

#     try:
#         _save_upload(file, temp_input_path)
#         result = transcriber.process_file(temp_input_path, temp_dir=UPLOAD_DIR)
#         _persist_result(result)
#         return JSONResponse(content=result)
#     except RuntimeError as exc:
#         raise HTTPException(status_code=500, detail=str(exc)) from exc
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc
#     finally:
#         try:
#             file.file.close()
#         except Exception:
#             pass


# # -- Async endpoint --------------------------------------------------------

# @app.post("/transcribe/async", response_model=JobSubmittedResponse, status_code=202)
# def transcribe_audio_async(file: UploadFile = File(...)):
#     """
#     Submit an audio file for background transcription.

#     Returns a ``job_id`` immediately (HTTP 202 Accepted).
#     Poll ``GET /jobs/{job_id}`` to retrieve the transcript once processing
#     is complete.  Large files are automatically split into chunks.
#     """
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="Missing filename.")

#     _validate_extension(file.filename)

#     # Save the upload synchronously before handing off to the background thread,
#     # because the UploadFile handle is only valid during this request lifecycle.
#     temp_input_path = os.path.join(UPLOAD_DIR, file.filename)
#     try:
#         _save_upload(file, temp_input_path)
#     except Exception as exc:
#         raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
#     finally:
#         try:
#             file.file.close()
#         except Exception:
#             pass

#     job_id = transcriber.process_file_async(temp_input_path, temp_dir=UPLOAD_DIR)
#     return {"job_id": job_id, "status": JobStatus.PENDING}


# # -- Job polling -----------------------------------------------------------

# @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
# def get_job(job_id: str):
#     """Return the current status (and result, if done) for a background job."""
#     record = job_store.get(job_id)
#     if record is None:
#         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

#     # Persist result to disk the first time we see it complete
#     if record["status"] == JobStatus.DONE and record.get("result"):
#         out_path = os.path.join(OUTPUT_DIR, f"{job_id}.json")
#         if not os.path.exists(out_path):
#             _persist_result(record["result"])

#     return {
#         "job_id": record["job_id"],
#         "status": record["status"],
#         "result": record.get("result"),
#         "error":  record.get("error"),
#     }


# @app.get("/jobs", response_model=list[JobStatusResponse])
# def list_jobs():
#     """Return a summary of all known jobs (useful for debugging / admin UIs)."""
#     return [
#         {
#             "job_id": r["job_id"],
#             "status": r["status"],
#             "result": r.get("result"),
#             "error":  r.get("error"),
#         }
#         for r in job_store.all()
#     ]


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