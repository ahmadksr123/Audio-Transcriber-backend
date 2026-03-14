


from pydantic import BaseModel
from typing import List, Optional
from transcriber import JobStatus


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class TranscriptMetadata(BaseModel):
    model: str
    source: str
    device: str
    chunks: int = 1     # number of audio chunks processed (1 = no splitting needed)


class TranscriptResponse(BaseModel):
    job_id: str
    language: Optional[str] = None
    text: str
    segments: List[TranscriptSegment]
    metadata: TranscriptMetadata


# ---------------------------------------------------------------------------
# Async job polling schemas
# ---------------------------------------------------------------------------

class JobSubmittedResponse(BaseModel):
    """Returned immediately when an async transcription job is accepted."""
    job_id: str
    status: JobStatus


class JobStatusResponse(BaseModel):
    """Returned when a client polls GET /jobs/{job_id}."""
    job_id: str
    status: JobStatus
    result: Optional[TranscriptResponse] = None
    error: Optional[str] = None


