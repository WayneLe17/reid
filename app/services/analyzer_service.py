import cv2
import tempfile
import os
from enum import Enum
from typing import List, Dict, Optional, Set
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

class VideoAnalysisResult(BaseModel):
    object_behaviors: List[ObjectBehavior] = Field(description="Behavior analysis for each tracked object")
    class_activity: str = Field(description="The primary activity of the class")

class AnalyzerService:
    def __init__(self):
        self.model_name = settings.GEMINI_MODEL
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY)
    
    def analyze_video_with_objects(self, video_path: str, object_ids: Set[int]) -> VideoAnalysisResult:
        try:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            
            action_values = [action.value for action in ActionType]
            prompt = f"""
            Analyze this classroom video with tracked students (bounding boxes with ID numbers).
            
            Video contains {len(object_ids)} tracked persons with IDs: {sorted(object_ids)}
            
            For EACH tracked person, determine their PRIMARY action throughout the video.
            Choose from: {', '.join(action_values)}
            
            Also determine the overall class activity.
            """
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=types.Content(
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                data=video_bytes,
                                mime_type='video/mp4'
                            )
                        ),
                        types.Part(text=prompt)
                    ]
                ),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VideoAnalysisResult
                )
            )
            
            return response.parsed
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return VideoAnalysisResult(
                object_behaviors=[
                    ObjectBehavior(object_id=oid, primary_action=ActionType.INACTIVE)
                    for oid in sorted(object_ids)
                ],
                class_activity="unknown"
            )
    
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