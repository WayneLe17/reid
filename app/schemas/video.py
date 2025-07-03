from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoProcessingRequest(BaseModel):
    tracking_method: str = Field(default="botsort", description="Tracking method to use")
    reid_model: str = Field(default="osnet_ibn_x1_0_msmt17.pt", description="ReID model path")
    yolo_model: str = Field(default="yolo11s.pt", description="YOLO model path")
    conf: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou: float = Field(default=0.7, ge=0.0, le=1.0, description="IoU threshold")
    distance_threshold: float = Field(default=0.2, ge=0.0, le=1.0, description="Clustering distance threshold")
    n_clusters: Optional[int] = Field(default=None, description="Number of clusters")
    vid_stride: int = Field(default=1, ge=1, description="Video stride")
    enable_visualization: bool = Field(default=True, description="Enable visualization")

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