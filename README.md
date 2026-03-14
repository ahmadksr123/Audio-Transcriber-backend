# Volga Partners — Audio Transcription Platform

CPU-based audio transcription service built with faster-whisper, FastAPI, and React.

---

## Stack

| Layer | Tech |
|---|---|
| Frontend | React (Vite) |
| Backend | FastAPI + faster-whisper |
| Audio processing | ffmpeg / ffprobe |
| Inference | CTranslate2 (int8, CPU) |

---

## Project Structure

```
backend/
├── app.py              # FastAPI routes, CORS, middleware
├── transcriber.py      # Chunking, Whisper inference, JobStore
├── schemas.py          # Pydantic models
├── requirements.txt    # Pinned deps (Python 3.11 wheels only)
├── render.yaml         # Render deploy config
├── .python-version     # Locks Python 3.11
├── temp/               # Uploaded audio + WAV chunks (auto-deleted)
└── outputs/            # Completed transcript JSON (keyed by job_id)

frontend/transcribe-app/
├── public/
│   └── index.html
├── src/
│   ├── App.jsx         # Upload UI, polling, results display
│   ├── index.js
│   └── index.css
├── package.json
└── README.md
```

---

## System Architecture

```
┌─────────────────────────────────┐
│         React Frontend          │
│  • Reads file → ArrayBuffer     │
│  • POST /transcribe             │
│  • Polls /jobs/{job_id}         │
└────────────────┬────────────────┘
                 │ HTTP multipart/form-data
┌────────────────▼────────────────┐
│        FastAPI  (app.py)        │
│  POST /transcribe               │
│  POST /transcribe/async         │
│  GET  /jobs/{job_id}            │
│  GET  /jobs                     │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   AudioTranscriber              │
│   (transcriber.py)              │
│                                 │
│  preprocess → split → detect    │
│  language → transcribe chunks   │
│  → dedup → merge → cleanup      │
└────────────────┬────────────────┘
                 │
┌────────────────▼────────────────┐
│   JobStore  (in-memory)         │
│   job_id → { status, progress,  │
│              result, error }    │
└─────────────────────────────────┘
```

---

## Request Flow

### Synchronous
```
Client                        FastAPI
  │                              │
  │  POST /transcribe ─────────► │
  │                              │  validate → save → preprocess
  │                              │  split → transcribe → merge
  │  ◄── 200 { transcript }      │  persist JSON → return
```

### Asynchronous (large files)
```
Client                  FastAPI              Background Thread
  │                        │                        │
  │  POST /async ────────► │                        │
  │                        │  save file             │
  │                        │  create job ─────────► │
  │  ◄── 202 { job_id }    │                        │  preprocess
  │                        │                        │  split chunks
  │  GET /jobs/{id} ─────► │                        │  detect language
  │  ◄── progress: 30%     │ ◄── store.update ────── │  transcribe chunk 1
  │                        │                        │  transcribe chunk 2
  │  GET /jobs/{id} ─────► │                        │  merge + dedup
  │  ◄── status: done      │ ◄── store.update ────── │  delete temp files
  │      result: {...}     │
```

---

## Transcription Pipeline

```
Input file (any format)
      │
      ▼
 preprocess_audio()
 ffmpeg → mono 16kHz WAV
 bandpass 80–8000 Hz
      │
      ▼
 _get_duration()  ──── ≤ 60s? ──── transcribe directly
      │
    > 60s
      │
      ▼
 _split_into_chunks()
 concurrent ffmpeg Popen
 60s chunks + 0.5s overlap
      │
      ▼
 _detect_language()
 Whisper on first chunk only
      │
      ▼
 _transcribe_single() × N chunks
 beam_size=1, VAD filter, no word timestamps
 timestamps shifted by chunk offset
      │
      ▼
 _dedup_segments()
 sort by start time
 remove overlap duplicates
      │
      ▼
 { job_id, language, text, segments[], metadata }
      │
      ▼
 outputs/{job_id}.json
```

---

## Data Models

```
TranscriptSegment
  start     float
  end       float
  text      str

TranscriptMetadata
  model         str
  source        str
  device        str
  chunks        int
  inter_threads int
  intra_threads int

TranscriptResponse
  job_id    str
  language  str?
  text      str
  segments  TranscriptSegment[]
  metadata  TranscriptMetadata

JobSubmittedResponse
  job_id    str
  status    pending | processing | done | failed

JobStatusResponse
  job_id    str
  status    pending | processing | done | failed
  progress  int  (0–100, real chunk-level)
  result    TranscriptResponse?
  error     str?
```

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/transcribe` | Synchronous — blocks until done |
| `POST` | `/transcribe/async` | Returns `job_id` immediately (202) |
| `GET` | `/jobs/{job_id}` | Poll status + progress + result |
| `GET` | `/jobs` | List all jobs |
<!-- it gives all the output folder stored json data -->

---

## Local Setup

**Requirements:** Python 3.11, Node 18+, ffmpeg

```bash
# Backend
cd backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Frontend
cd frontend/transcribe-app
npm install && npm start
```

---

## Render Deployment

| Field | Value |
|---|---|
| Python Version | `3.11` |
| Build Command | `pip install --upgrade pip && pip install -r requirements.txt` |
| Start Command | `uvicorn app:app --host 0.0.0.0 --port $PORT` |

> ⚠️ `ffmpeg` must be added as a native package via `render.yaml` — not available in the Render dashboard UI.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `small` | `tiny` / `base` / `small` / `medium` |
| `FRONTEND_URL` | — | Added to CORS allow-list |
| `OMP_NUM_THREADS` | `2` | CTranslate2 thread cap |

---

## Performance (2-core CPU, model=small)

| Audio | Time |
|---|---|
| 2 min | ~25 s |
| 10 min | ~2 min |
| 30 min | ~6 min |

Use `WHISPER_MODEL=tiny` to halve times at ~5% accuracy cost.
