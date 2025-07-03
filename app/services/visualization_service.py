import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class VisualizationService:
    def __init__(self):
        self.cluster_map = {}
        self.unified_data = None

    def get_cluster_mapping(self, cluster_results: Dict):
        self.cluster_map = {}
        if not cluster_results or 'clusters' not in cluster_results:
            return
        
        for cluster_id, tracking_ids in cluster_results['clusters'].items():
            for tracking_id in tracking_ids:
                self.cluster_map[tracking_id] = cluster_id

    def get_cluster_color(self, cluster_id: int) -> Tuple[int, int, int]:
        np.random.seed(cluster_id * 42)
        return tuple(np.random.randint(50, 255, 3).tolist())

    def load_unified_data(self, tracking_results_dir: str) -> bool:
        unified_file = Path(tracking_results_dir) / "tracking_results.json"
        if unified_file.exists():
            with open(unified_file, 'r') as f:
                self.unified_data = json.load(f)
            return True
        return False

    def load_frame_tracking_data(self, tracking_results_dir: str, frame_num: int) -> Optional[Dict]:
        if self.unified_data:
            for frame_data in self.unified_data.get("frames", []):
                if frame_data.get("frame_number") == frame_num:
                    return frame_data
            return None
        else:
            frame_file = Path(tracking_results_dir) / f"frame_{frame_num:06d}.json"
            if frame_file.exists():
                with open(frame_file, 'r') as f:
                    return json.load(f)
            return None

    def draw_bbox_with_cluster(self, frame: np.ndarray, bbox: List[float], 
                              tracking_id: int, cluster_id: Optional[int] = None) -> np.ndarray:
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        if cluster_id is not None:
            color = self.get_cluster_color(cluster_id)
            label = f"Track:{tracking_id} Cluster:{cluster_id}"
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

    def create_cluster_legend(self, frame: np.ndarray, cluster_results: Dict) -> np.ndarray:
        if not cluster_results or 'clusters' not in cluster_results:
            return frame
        
        legend_x = frame.shape[1] - 300
        legend_y = 30
        
        legend_height = len(cluster_results['clusters']) * 25 + 20
        cv2.rectangle(frame, 
                     (legend_x - 10, legend_y - 10),
                     (frame.shape[1] - 10, legend_y + legend_height),
                     (0, 0, 0), -1)
        
        cv2.putText(frame, "Cluster Legend:",
                   (legend_x, legend_y + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                   (255, 255, 255), 2)
        
        y_offset = 35
        for cluster_id in sorted(cluster_results['clusters'].keys()):
            color = self.get_cluster_color(cluster_id)
            tracking_ids = cluster_results['clusters'][cluster_id]
            
            cv2.rectangle(frame,
                         (legend_x, legend_y + y_offset - 8),
                         (legend_x + 15, legend_y + y_offset + 8),
                         color, -1)
            
            text = f"C{cluster_id}: {len(tracking_ids)} IDs"
            cv2.putText(frame, text,
                       (legend_x + 20, legend_y + y_offset + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 255), 1)
            
            y_offset += 25
        
        return frame

    def process_video(self, video_path: str, tracking_results_dir: str, 
                     output_path: str, cluster_results: Optional[Dict] = None) -> Dict:
        if cluster_results:
            self.get_cluster_mapping(cluster_results)
        
        self.load_unified_data(tracking_results_dir)
        
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
            
            frame_data = self.load_frame_tracking_data(tracking_results_dir, frame_num)
            
            if frame_data and 'detections' in frame_data:
                for detection in frame_data['detections']:
                    tracking_id = detection['track_id']
                    bbox = detection['bbox']
                    
                    cluster_id = self.cluster_map.get(tracking_id)
                    frame = self.draw_bbox_with_cluster(frame, bbox, tracking_id, cluster_id)
            
            if cluster_results:
                frame = self.create_cluster_legend(frame, cluster_results)
            
            frame_info = f"Frame: {frame_num}/{total_frames-1}"
            cv2.putText(frame, frame_info,
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (255, 255, 255), 2)
            
            out.write(frame)
            processed_frames += 1
            frame_num += 1
        
        cap.release()
        out.release()
        
        return {
            'input_path': video_path,
            'output_path': output_path,
            'processed_frames': processed_frames,
            'cluster_mappings': len(self.cluster_map) if cluster_results else 0
        }