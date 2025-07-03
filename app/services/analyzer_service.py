import cv2
import tempfile
import os
from enum import Enum
from typing import List, Dict, Optional, Set
from pathlib import Path
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from app.core.config import settings

class ActionType(str, Enum):
    SITTING = "sitting"
    STANDING = "standing"
    WRITING = "writing"
    READING = "reading"
    RAISING_HAND = "raising_hand"
    TALKING = "talking"
    LOOKING_AT_BOARD = "looking_at_board"
    WALKING = "walking"
    GROUP_DISCUSSION = "group_discussion"
    INACTIVE = "inactive"
    USING_PHONE = "using_phone"
    SLEEPING = "sleeping"
    EATING = "eating"
    LEAVING_SEAT = "leaving_seat"
    ENTERING_CLASS = "entering_class"
    TAKING_NOTES = "taking_notes"
    LISTENING = "listening"
    PRESENTING = "presenting"

class ObjectBehavior(BaseModel):
    object_id: int = Field(description="The ID of the tracked person")
    primary_action: ActionType = Field(description="The primary action of this person")

class ChunkAnalysisResult(BaseModel):
    chunk_number: int = Field(description="The chunk number")
    start_frame: int = Field(description="Starting frame for this chunk")
    end_frame: int = Field(description="Ending frame for this chunk")
    object_behaviors: List[ObjectBehavior] = Field(description="Behavior analysis for each tracked object in this chunk")
    class_activity: str = Field(description="The activity of the class in this chunk")

class VideoAnalysisResult(BaseModel):
    chunk_results: List[ChunkAnalysisResult] = Field(description="Analysis results for each chunk")

class AnalyzerService:
    def __init__(self):
        self.model_name = settings.GEMINI_MODEL
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    
    def create_video_chunk(self, video_path: str, start_frame: int, end_frame: int) -> str:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for _ in range(start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        
        return temp_path
    
    def analyze_single_chunk(self, crops_dir: str, chunk_number: int, cluster_results: Dict) -> Dict:
        if 'clusters' not in cluster_results:
            return {"behaviors": {}, "class_activity": "unknown"}
        
        behaviors = {}
        cluster_ids = set(int(cluster_id) for cluster_id in cluster_results['clusters'].keys())
        for cluster_id in cluster_ids:
            tracking_ids = cluster_results['clusters'].get(cluster_id, [])
            
            chunk_crop_path = None
            for tracking_id in tracking_ids:
                potential_path = Path(crops_dir) / f"id_{tracking_id}" / f"chunk_{chunk_number}.jpg"
                if potential_path.exists():
                    chunk_crop_path = potential_path
                    break
            
            behaviors[cluster_id] = self.analyze_single_crop(chunk_crop_path, cluster_id)
        
        class_activity = self.analyze_single_frame_activity(crops_dir, chunk_number)
        
        return {
            "behaviors": behaviors,
            "class_activity": class_activity
        }
    
    def analyze_single_crop(self, crop_path: Path, cluster_id: int) -> ActionType:
        try:
            with open(crop_path, 'rb') as f:
                crop_data = f.read()
            
            action_values = [action.value for action in ActionType]
            prompt = f"""
            Analyze this cropped image of person from cluster {cluster_id} in a classroom.
            
            Determine what action this person is performing.
            Choose from: {', '.join(action_values)}
            
            Return only the action name, nothing else.
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=types.Content(parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=crop_data,
                            mime_type='image/jpeg'
                        )
                    ),
                    types.Part(text=prompt)
                ]),
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=50
                )
            )
            
            action_text = response.text.strip().lower()
            for action in ActionType:
                if action.value.lower() in action_text:
                    return action
            
            return ActionType.INACTIVE
            
        except Exception as e:
            print(f"Error analyzing crop for cluster {cluster_id}: {e}")
            return ActionType.INACTIVE
    
    def analyze_single_frame_activity(self, crops_dir: str, chunk_number: int) -> str:
        frame_path = Path(crops_dir) / "chunk_frames" / f"frame_chunk_{chunk_number}.jpg"
        
        if not frame_path.exists():
            return "unknown"
        
        try:
            with open(frame_path, 'rb') as f:
                frame_data = f.read()
            
            prompt = f"""
            Analyze this classroom frame.
            
            Determine the PRIMARY ACTIVITY happening in this classroom at this moment.
            
            Examples: lecture, discussion, individual work, test, lab work, break time
            
            Return only the activity name, nothing else.
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=types.Content(parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=frame_data,
                            mime_type='image/jpeg'
                        )
                    ),
                    types.Part(text=prompt)
                ]),
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=50
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Error analyzing frame activity for chunk {chunk_number}: {e}")
            return "unknown"
    
    def analyze_cluster_behaviors_from_frames(self, crops_dir: str, cluster_results: Dict) -> Optional[VideoAnalysisResult]:
        if not cluster_results or 'clusters' not in cluster_results:
            return None
        
        cluster_ids = set(int(cluster_id) for cluster_id in cluster_results['clusters'].keys())
        
        if not cluster_ids:
            return None
        
        frames_folder = Path(crops_dir) / "chunk_frames"
        if not frames_folder.exists():
            return None
        
        chunk_files = list(frames_folder.glob("frame_chunk_*.jpg"))
        if not chunk_files:
            return None
        
        chunk_numbers = [int(f.stem.split('_')[2]) for f in chunk_files]
        chunk_numbers.sort()
        
        chunk_interval_frames = int(settings.ANALYSIS_CHUNK_MINUTES * 60 * 30)  # Assuming 30 FPS, should get from video
        
        chunk_results = []
        
        for chunk_number in chunk_numbers:
            chunk_result = self.analyze_single_chunk(crops_dir, chunk_number, cluster_results)
            
            chunk_behaviors = []
            for cluster_id in sorted(cluster_ids):
                action = chunk_result["behaviors"].get(cluster_id, ActionType.INACTIVE)
                chunk_behaviors.append(ObjectBehavior(
                    object_id=cluster_id,
                    primary_action=action
                ))
            
            start_frame = chunk_number * chunk_interval_frames
            end_frame = (chunk_number + 1) * chunk_interval_frames
            
            chunk_results.append(ChunkAnalysisResult(
                chunk_number=chunk_number,
                start_frame=start_frame,
                end_frame=end_frame,
                object_behaviors=chunk_behaviors,
                class_activity=chunk_result["class_activity"]
            ))
        
        return VideoAnalysisResult(
            chunk_results=chunk_results
        )
    
    def analyze_cluster_behaviors(self, video_path: str, cluster_results: Dict) -> Optional[VideoAnalysisResult]:
        if not cluster_results or 'clusters' not in cluster_results:
            return None
        
        cluster_ids = set(int(cluster_id) for cluster_id in cluster_results['clusters'].keys())
        
        if not cluster_ids:
            return None
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        chunk_frames = int(fps * 60 * settings.ANALYSIS_CHUNK_MINUTES)
        
        if total_frames <= chunk_frames:
            return self.analyze_video_with_objects(video_path, cluster_ids)
        
        behaviors_per_cluster = {}
        
        for start_frame in range(0, total_frames, chunk_frames):
            end_frame = min(start_frame + chunk_frames, total_frames)
            
            chunk_path = self.create_video_chunk(video_path, start_frame, end_frame)
            
            try:
                chunk_result = self.analyze_video_with_objects(chunk_path, cluster_ids)
                
                for behavior in chunk_result.object_behaviors:
                    cluster_id = behavior.object_id
                    action = behavior.primary_action
                    
                    if cluster_id not in behaviors_per_cluster:
                        behaviors_per_cluster[cluster_id] = []
                    behaviors_per_cluster[cluster_id].append(action)
                
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
        
        final_behaviors = []
        for cluster_id in sorted(cluster_ids):
            if cluster_id in behaviors_per_cluster:
                actions = behaviors_per_cluster[cluster_id]
                most_common_action = max(set(actions), key=actions.count)
                final_behaviors.append(ObjectBehavior(
                    object_id=cluster_id,
                    primary_action=most_common_action
                ))
            else:
                final_behaviors.append(ObjectBehavior(
                    object_id=cluster_id,
                    primary_action=ActionType.INACTIVE
                ))
        
        return VideoAnalysisResult(
            object_behaviors=final_behaviors,
            class_activity="mixed activities"
        )