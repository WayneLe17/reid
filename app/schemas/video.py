from typing import Optional, Dict, Any
from pydantic import BaseModel
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class VideoProcessingResponse(BaseModel):
    task_id: str
    status: ProcessingStatus
    message: str
    progress: Optional[float] = None
    output_path: Optional[str] = None
    created_at: str
    updated_at: str

class VideoProcessingResult(BaseModel):
    task_id: str
    status: ProcessingStatus
    output_video_path: Optional[str] = None
    tracking_results: Optional[Dict[str, Any]] = None
    cluster_results: Optional[Dict[str, Any]] = None
    processing_stats: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None

class TaskStatus(BaseModel):
    task_id: str
    status: ProcessingStatus
    progress: float
    message: str
    created_at: str
    updated_at: str