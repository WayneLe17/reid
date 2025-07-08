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
        if not cluster_results:
            return {}
            
        if 'tracking_to_cluster' in cluster_results:
            return cluster_results['tracking_to_cluster']
        
        cluster_map = {}
        for cluster_id, tracking_ids in cluster_results.get('clusters', {}).items():
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
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        if cluster_id is not None:
            color = self.get_cluster_color(cluster_id)
            label = f"ID:{cluster_id}"
            if action:
                label += f" {action.activity.value}"
        else:
            color = (0, 255, 0)  # Green color like in test.py
            label = f"ID:{tracking_id}"
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background and text
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(frame, 
                     (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1),
                     color, -1)
        
        cv2.putText(frame, label,
                   (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 2)
        
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
        
        frame_width = frame.shape[1]
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
    
    def export_frame_data_to_json(self, output_path: str, tracking_results_dir: str,
                                  cluster_results: Optional[Dict] = None,
                                  analysis_results: Optional[VideoAnalysisResult] = None) -> str:
        cluster_map = self.get_cluster_mapping(cluster_results)
        frames_tracking_data = self.load_frame_tracking_data(tracking_results_dir)
        
        if not frames_tracking_data:
            raise ValueError("No tracking data found")
        
        export_data = {
            "video_info": frames_tracking_data.get("video_info", {}),
            "frames": []
        }
        
        # Sort frames by frame number to maintain order
        sorted_frames = sorted(frames_tracking_data.get("frames", {}).items(), key=lambda x: int(x[0]))
        for frame_num_str, frame_data in sorted_frames:
            frame_num = int(frame_num_str)  # Convert string key to int
            fps = frames_tracking_data.get("video_info", {}).get("fps", 30)
            
            chunk_number = self.get_chunk_for_frame(frame_num, fps)
            behavior_map = self.get_behaviors_for_chunk(chunk_number, analysis_results)
            class_activity = self.get_class_activity_for_chunk(chunk_number, analysis_results)
            
            frame_export = {
                "frame_number": frame_num,
                "timestamp": frame_data.get("timestamp", ""),
                "class_activity": class_activity,
                "detections": []
            }
            
            for detection in frame_data.get("detections", []):
                tracking_id = detection["track_id"]
                bbox = detection["bbox"]
                cluster_id = cluster_map.get(tracking_id)
                action = behavior_map.get(cluster_id) if cluster_id else None
                
                detection_export = {
                    "track_id": tracking_id,
                    "cluster_id": cluster_id,
                    "bbox": {
                        "x": bbox[0],
                        "y": bbox[1], 
                        "width": bbox[2],
                        "height": bbox[3]
                    },
                    "action_type": None
                }
                
                if action:
                    detection_export["action_type"] = {
                        "activity": action.activity.value,
                        "posture": action.posture.value,
                        "focus_level": action.focus_level.value,
                        "unusual_behaviors": action.unusual_behaviors
                    }
                
                frame_export["detections"].append(detection_export)
            
            export_data["frames"].append(frame_export)
        
        json_output_path = output_path.replace(".mp4", "_analysis_results.json")
        with open(json_output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return json_output_path

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
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_num = 0
        processed_frames = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Starting video processing: {total_frames} frames total")
        if frames_tracking_data:
            print(f"Loaded tracking data with {len(frames_tracking_data.get('frames', {}))} frames")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            chunk_number = self.get_chunk_for_frame(frame_num, fps)
            behavior_map = self.get_behaviors_for_chunk(chunk_number, analysis_results)
            class_activity = self.get_class_activity_for_chunk(chunk_number, analysis_results)
            
            frame_key = str(frame_num)
            frame_data = frames_tracking_data.get('frames', {}).get(frame_key)
            
            if frame_data and 'detections' in frame_data:
                detections = frame_data['detections']
                print(f"Frame {frame_num}: Found {len(detections)} detections")
                for detection in detections:
                    tracking_id = detection['track_id']
                    bbox = detection['bbox']
                    
                    cluster_id = cluster_map.get(tracking_id)
                    action = behavior_map.get(cluster_id) if cluster_id else None
                    frame = self.draw_bbox_with_cluster(frame, bbox, tracking_id, cluster_id, action)
            elif frame_num % 100 == 0:  # Print every 100 frames for debugging
                print(f"Frame {frame_num}: No detections found")
            
            frame = self.draw_class_activity(frame, class_activity)
            out.write(frame)
            processed_frames += 1
            frame_num += 1
        
        cap.release()
        out.release()
        
        # Export frame data to JSON
        json_output_path = self.export_frame_data_to_json(
            output_path, tracking_results_dir, cluster_results, analysis_results
        )
        
        return {
            'input_path': video_path,
            'output_path': output_path,
            'json_output_path': json_output_path
        }