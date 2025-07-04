import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from app.core.config import settings
from app.services.analyzer_service import VideoAnalysisResult, ActionType

class VisualizationService:
    def __init__(self):
        pass

    def get_cluster_mapping(self, cluster_results: Dict):
        if 'tracking_to_cluster' in cluster_results:
            return cluster_results['tracking_to_cluster']
        
        cluster_map = {}
        for cluster_id, tracking_ids in cluster_results['clusters'].items():
            for tracking_id in tracking_ids:
                cluster_map[tracking_id] = int(cluster_id)
        return cluster_map

    def get_cluster_color(self, cluster_id: int) -> Tuple[int, int, int]:
        np.random.seed(cluster_id * 42)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def load_frame_tracking_data(self, tracking_results_dir: str) -> Optional[Dict] :
        frames_tracking_path = Path(tracking_results_dir) / "tracking_results.json"
        if frames_tracking_path.exists():
            with open(frames_tracking_path, 'r') as f:
                frames_tracking_data = json.load(f)
            return frames_tracking_data
        return None

    def draw_bbox_with_cluster(self, frame: np.ndarray, bbox: List[float], 
                              tracking_id: int, cluster_id: Optional[int] = None,
                              action: Optional[ActionType] = None) -> np.ndarray:
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        if cluster_id is not None:
            color = self.get_cluster_color(cluster_id)
            label = f"ID{cluster_id}\n"
            if action:
                label += f"Activity: {action.activity.value}\n"
                label += f"Posture: {action.posture.value}\n"
                label += f"Focus Level: {action.focus_level.value}\n"
                label += f"Unusual Behaviors: {action.unusual_behaviors}"
        else:
            color = (128, 128, 128)
            label = f"Track:{tracking_id}"
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, 
                     (int(x1), int(y1) - label_size[1] - 10),
                     (int(x1) + label_size[0], int(y1)),
                     color, -1)
        
        cv2.putText(frame, label,
                   (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)
        
        return frame


    def create_behavior_legend(self, frame: np.ndarray, behavior_map: Dict[int, str]) -> np.ndarray:
        if not behavior_map:
            return frame
        
        legend_x = 20
        legend_y = frame.shape[0] - 200
        legend_width = 300
        legend_height = len(behavior_map) * 25 + 40
        
        cv2.rectangle(frame, 
                     (legend_x - 10, legend_y - 30),
                     (legend_x + legend_width, legend_y + legend_height - 30),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, "Behavior Map:",
                   (legend_x, legend_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)
        
        y_offset = 0
        for cluster_id, behavior in sorted(behavior_map.items()):
            color = self.get_cluster_color(cluster_id)
            
            cv2.rectangle(frame,
                         (legend_x, legend_y + y_offset),
                         (legend_x + 20, legend_y + y_offset + 20),
                         color, -1)
            
            text = f"ID {cluster_id}: {behavior}"
            cv2.putText(frame, text,
                       (legend_x + 30, legend_y + y_offset + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                       (255, 255, 255), 1)
            
            y_offset += 25
        
        return frame

    def get_chunk_for_frame(self, frame_num: int, fps: float) -> int:
        chunk_interval_frames = int(fps * 60 * settings.ANALYSIS_CHUNK_MINUTES)
        return frame_num // chunk_interval_frames
    
    def get_behaviors_for_chunk(self, chunk_number: int, analysis_results: VideoAnalysisResult) -> Dict[int, ActionType]:
        if not analysis_results or not hasattr(analysis_results, 'chunk_results'):
            return {}
        
        for chunk_result in analysis_results.chunk_results:
            if chunk_result.chunk_number == chunk_number:
                return {
                    behavior.object_id: behavior.primary_action 
                    for behavior in chunk_result.object_behaviors
                }
        return {}
    
    def get_class_activity_for_chunk(self, chunk_number: int, analysis_results) -> str:
        if not analysis_results or not hasattr(analysis_results, 'chunk_results'):
            return "unknown"
        
        for chunk_result in analysis_results.chunk_results:
            if chunk_result.chunk_number == chunk_number:
                return chunk_result.class_activity
        return "unknown"
    
    def draw_class_activity(self, frame: np.ndarray, class_activity: str) -> np.ndarray:
        if not class_activity or class_activity == "unknown":
            return frame
        
        frame_height, frame_width = frame.shape[:2]
        text = f"Class Activity: {class_activity}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        text_x = frame_width - text_size[0] - 20
        text_y = 40
        
        cv2.rectangle(frame, 
                     (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, text,
                   (text_x, text_y),
                   font, font_scale,
                   (255, 255, 255), thickness)
        
        return frame
    
    def process_video(self, video_path: str, tracking_results_dir: str, 
                     output_path: str, cluster_results: Optional[Dict] = None,
                     analysis_results: Optional[VideoAnalysisResult] = None) -> Dict:
        cluster_map = self.get_cluster_mapping(cluster_results)
        
        frames_tracking_data = self.load_frame_tracking_data(tracking_results_dir)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        processed_frames = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            chunk_number = self.get_chunk_for_frame(frame_num, fps)
            behavior_map = self.get_behaviors_for_chunk(chunk_number, analysis_results)
            class_activity = self.get_class_activity_for_chunk(chunk_number, analysis_results)
            
            frame_data = frames_tracking_data.get('frames', {})[frame_num]
            
            if frame_data and 'detections' in frame_data:
                for detection in frame_data['detections']:
                    tracking_id = detection['track_id']
                    bbox = detection['bbox']
                    
                    cluster_id = cluster_map.get(tracking_id)
                    action = behavior_map.get(cluster_id) if cluster_id else None
                    frame = self.draw_bbox_with_cluster(frame, bbox, tracking_id, cluster_id, action)
            
            frame = self.draw_class_activity(frame, class_activity)
            out.write(frame)
            processed_frames += 1
            frame_num += 1
        
        cap.release()
        out.release()
        
        return {
            'input_path': video_path,
            'output_path': output_path,
        }