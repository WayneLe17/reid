import os
import json
import time
import cv2
from typing import Dict, List, Optional
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import TRACKER_CONFIGS
from boxmot.engine.detectors import is_ultralytics_model
from ultralytics import YOLO
from ultralytics.utils import plotting
from pathlib import Path
    
plotting.Annotator.box = lambda *args, **kwargs: None
plotting.Annotator.box_label = lambda *args, **kwargs: None
plotting.Annotator.line = lambda *args, **kwargs: None

class TrackingService:
    def __init__(self):
        self.fps = None
        self.frame_count = 0
        self.start_time = None
        self.seen_ids = set()
        self.frame_results = []
        self.id_crops = {}

    def set_fps(self, fps: float):
        self.fps = fps

    def frame_to_timestamp(self, frame_num: int) -> str:
        if self.fps is None or self.fps == 0:
            return f"frame_{frame_num}"
        seconds = frame_num / self.fps
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def save_first_crop_only(self, track_id: int, bbox: List[float], frame_img):
        if track_id in self.seen_ids:
            return
            
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        img_h, img_w = frame_img.shape[:2]
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))
        
        if x2 > x1 and y2 > y1:
            self.id_crops[track_id] = {
                'image': frame_img[y1:y2, x1:x2].copy(),
            }
            self.seen_ids.add(track_id)

    def start_timing(self):
        self.start_time = time.time()

    def record_frame_time(self):
        pass

    def get_final_stats(self) -> Optional[Dict]:
        if self.start_time and self.frame_count > 0:
            total_time = time.time() - self.start_time
            avg_fps = self.frame_count / total_time
            return {
                "total_frames": self.frame_count,
                "total_time": f"{total_time:.2f}s",
                "average_fps": f"{avg_fps:.2f}",
                "unique_ids": len(self.seen_ids),
                "crops_in_memory": len(self.id_crops)
            }
        return None

    def save_memory_to_disk(self, crops_dir: str, results_dir: str):
        if crops_dir and self.id_crops:
            os.makedirs(crops_dir, exist_ok=True)
            
            for track_id, crop_data in self.id_crops.items():
                id_folder = os.path.join(crops_dir, f"id_{track_id}")
                os.makedirs(id_folder, exist_ok=True)
                
                crop_filename = "crop.jpg"
                crop_path = os.path.join(id_folder, crop_filename)
                cv2.imwrite(crop_path, crop_data['image'])
        
        if results_dir and self.frame_results:
            os.makedirs(results_dir, exist_ok=True)
            
            all_results = {
                "video_info": {
                    "total_frames": len(self.frame_results),
                    "fps": self.fps,
                    "processing_stats": self.get_final_stats()
                },
                "frames": self.frame_results
            }
            
            results_file = os.path.join(results_dir, "tracking_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, separators=(',', ':'))

    def process_video(self, video_path: str, yolo_model: str = 'yolo11s.pt', 
                     tracking_method: str = 'botsort', reid_model: str = 'osnet_ibn_x1_0_msmt17.pt',
                     conf: float = 0.5, iou: float = 0.7, imgsz: int = 640,
                     device: int = 0, classes: List[int] = [0], vid_stride: int = 1,
                     crops_dir: str = None, results_dir: str = None,
                    ) -> Dict:
        
        self.frame_count = 0
        self.start_time = None
        self.seen_ids = set()
        self.frame_results = []
        self.id_crops = {}
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        self.set_fps(fps)
        self.start_timing()
        
        tracking_config = TRACKER_CONFIGS / (tracking_method + '.yaml')
        tracker = create_tracker(
            tracking_method,
            tracking_config,
            Path(reid_model),
            device,
            False,
            False,
        )
        
        if hasattr(tracker, "model"):
            tracker.model.warmup()
        
        yolo = YOLO(yolo_model if is_ultralytics_model(yolo_model) else "yolov8n.pt")
        
        results = yolo.track(
            source=video_path,
            conf=conf,
            iou=iou,
            device=str(device),
            classes=classes,
            imgsz=imgsz,
            vid_stride=vid_stride,
            save=False,
            stream=True,
            verbose=True,
        )
        
        for result in results:
            frame_detections = []
            
            if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                if hasattr(boxes, 'id') and boxes.id is not None:
                    for box, track_id in zip(boxes.xyxy, boxes.id):
                        x1, y1, x2, y2 = box.tolist()
                        bbox = [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]
                        track_id_int = int(track_id.item())
                        
                        detection = {
                            "track_id": track_id_int,
                            "bbox": bbox
                        }
                        frame_detections.append(detection)
                        
                        self.save_first_crop_only(track_id_int, bbox, result.orig_img)
            
            if results_dir and frame_detections:
                self.frame_results.append({
                    "frame_number": self.frame_count,
                    "timestamp": self.frame_to_timestamp(self.frame_count),
                    "detections": frame_detections
                })
            
            self.frame_count += 1
            self.record_frame_time()
        
        self.save_memory_to_disk(crops_dir, results_dir)
        
        tracking_results = {}
        for result_data in self.frame_results:
            timestamp = result_data["timestamp"]
            tracking_results[timestamp] = result_data["detections"]
        
        return {
            'tracking_results': tracking_results,
            'stats': self.get_final_stats(),
            'crops_dir': crops_dir,
            'results_dir': results_dir
        }