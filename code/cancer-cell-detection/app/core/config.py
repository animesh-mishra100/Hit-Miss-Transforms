from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Cancer Cell Detection API"
    UPLOAD_DIR: Path = Path("static/uploads")
    
    class Config:
        case_sensitive = True

settings = Settings() 