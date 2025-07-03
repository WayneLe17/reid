import aiofiles
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from app.schemas.video import VideoProcessingResponse, VideoProcessingResult, TaskStatus, ProcessingStatus
from app.services.video_service import video_service
from app.core.config import settings
from datetime import datetime

router = APIRouter()

@router.post("/process", response_model=VideoProcessingResponse)
async def process_video(
    background_tasks: BackgroundTasks,
    file_name: str = Form(...),
    tracking_method: str = Form("botsort"),
    reid_model: str = Form("osnet_ibn_x1_0_msmt17.pt"),
    yolo_model: str = Form("yolo11s.pt"),
    conf: float = Form(0.5),
    iou: float = Form(0.7),
    distance_threshold: float = Form(0.2),
    n_clusters: int = Form(10),
    vid_stride: int = Form(1),
):
    if not file_name:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_extension = Path(file_name).suffix.lower()
    if file_extension not in settings.ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {settings.ALLOWED_VIDEO_EXTENSIONS}"
        )
    
    task_id = video_service.create_task()
    
    request_params = {
        'tracking_method': tracking_method,
        'reid_model': reid_model,
        'yolo_model': yolo_model,
        'conf': conf,
        'iou': iou,
        'distance_threshold': distance_threshold,
        'n_clusters': n_clusters,
        'vid_stride': vid_stride,
    }
    
    background_tasks.add_task(
        video_service.process_video,
        task_id,
        file_name,
        request_params
    )
    
    return VideoProcessingResponse(
        task_id=task_id,
        status=ProcessingStatus.PENDING,
        message="Video processing started",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat()
    )

@router.get("/status/{task_id}", response_model=VideoProcessingResult)
async def get_task_status(task_id: str):
    task = video_service.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.get("/download/{task_id}")
async def download_result(task_id: str):
    task = video_service.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status != ProcessingStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Task not completed")
    
    if not task.output_video_path or not Path(task.output_video_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=task.output_video_path,
        filename=f"processed_{task_id}.mp4",
        media_type="video/mp4"
    )

@router.delete("/cleanup/{task_id}")
async def cleanup_task(task_id: str):
    task = video_service.get_task_status(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    try:
        video_service.cleanup_task_files(task_id)
        
        if task_id in video_service.tasks:
            del video_service.tasks[task_id]
        
        return {"message": f"Task {task_id} cleaned up successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/tasks", response_model=List[TaskStatus])
async def list_tasks():
    tasks = []
    for task_id, task in video_service.tasks.items():
        tasks.append(TaskStatus(
            task_id=task_id,
            status=task.status,
            progress=0.0,
            message=task.error_message or "Processing",
            created_at=task.created_at,
            updated_at=task.completed_at or task.created_at
        ))
    return tasks