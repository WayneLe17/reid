import os
import uuid
import asyncio
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
from app.services.tracking_service import TrackingService
from app.services.reid_service import ReIDService
from app.services.visualization_service import VisualizationService
from app.services.analyzer_service import AnalyzerService
from app.schemas.video import ProcessingStatus, VideoProcessingResult
from app.core.config import settings

class VideoProcessingService:
    def __init__(self):
        self.tasks: Dict[str, VideoProcessingResult] = {}
        self.tracking_service = TrackingService()
        self.reid_service = ReIDService()
        self.visualization_service = VisualizationService()
        self.analyzer_service = AnalyzerService()

    def create_task(self) -> str:
        task_id = str(uuid.uuid4())
        self.tasks[task_id] = VideoProcessingResult(
            task_id=task_id,
            status=ProcessingStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        return task_id

    def get_task_status(self, task_id: str) -> Optional[VideoProcessingResult]:
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: ProcessingStatus, 
                          **kwargs):
        if task_id in self.tasks:
            task = self.tasks[task_id]
            task.status = status
            for key, value in kwargs.items():
                if hasattr(task, key):
                    setattr(task, key, value)

    async def process_video(self, task_id: str, video_path: str, 
                           request_params: Dict[str, Any]) -> VideoProcessingResult:
        try:
            self.update_task_status(task_id, ProcessingStatus.PROCESSING)
            
            # Create output directories
            task_output_dir = Path(settings.OUTPUT_DIR) / task_id
            crops_dir = task_output_dir / "crops"
            results_dir = task_output_dir / "tracking_results"
            
            os.makedirs(task_output_dir, exist_ok=True)
            os.makedirs(crops_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)
            
            # Step 1: Run tracking
            tracking_params = {
                'crops_dir': str(crops_dir),
                'results_dir': str(results_dir),
            }
            
            tracking_results = await asyncio.to_thread(
                self.tracking_service.process_video, **tracking_params
            )
            
            clustering_params = {
                'crops_dir': str(crops_dir),
                'n_clusters': request_params.get('n_clusters')
            }
            
            cluster_results = await asyncio.to_thread(
                self.reid_service.process_clustering, **clustering_params
            )
            
            analysis_results = await asyncio.to_thread(
                self.analyzer_service.analyze_cluster_behaviors_from_frames, 
                str(crops_dir), 
                cluster_results
            )
            
            output_video_path = str(task_output_dir / "output_video.mp4")
            
            visualization_params = {
                'video_path': video_path,
                'tracking_results_dir': str(results_dir),
                'output_path': output_video_path,
                'cluster_results': cluster_results,
                'analysis_results': analysis_results
            }
            
            visualization_results = await asyncio.to_thread(
                self.visualization_service.process_video, **visualization_params
            )
            
            self.update_task_status(
                task_id,
                ProcessingStatus.COMPLETED,
                output_video_path=output_video_path,
                json_results_path=visualization_results.get('json_output_path'),
                tracking_results=tracking_results,
                cluster_results=cluster_results,
                processing_stats={
                    'tracking_stats': tracking_results.get('stats'),
                    'analysis_results': analysis_results.model_dump() if analysis_results else None,
                    'visualization_stats': visualization_results if 'visualization_results' in locals() else None
                },
                completed_at=datetime.now().isoformat()
            )
            
            return self.tasks[task_id]
            
        except Exception as e:
            self.update_task_status(
                task_id,
                ProcessingStatus.FAILED,
                error_message=str(e)
            )
            raise e

    def cleanup_task_files(self, task_id: str):
        task_output_dir = Path(settings.OUTPUT_DIR) / task_id
        if task_output_dir.exists():
            shutil.rmtree(task_output_dir)

video_service = VideoProcessingService()