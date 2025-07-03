from typing import List
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    PROJECT_NAME: str = "ReID Video Processing API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    REID_MODEL_PATH: str = "osnet_ibn_x1_0_msmt17.pt"
    YOLO_MODEL_PATH: str = "yolo11s.pt"
    DEVICE: int = 0
    
    CROPS_DIR: str = "crops"
    TRACKING_RESULTS_DIR: str = "tracking_results_frames"
    OUTPUT_DIR: str = "outputs"
    
    ALLOWED_VIDEO_EXTENSIONS: List[str] = [".mp4", ".avi", ".mov", ".mkv"]
    
    GOOGLE_API_KEY: str = ""
    GEMINI_MODEL: str = "models/gemini-2.0-flash"
    ANALYSIS_CHUNK_MINUTES: int = 1
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()