import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Traverse upwards to find and load .env file
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir and _current_dir != "/":
    _env_path = os.path.join(_current_dir, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break
    _current_dir = os.path.dirname(_current_dir)
else:
    load_dotenv()

class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "NLP Research Analyzer"
    
    # Security
    SECRET_KEY: str = os.getenv("SECRET_KEY", "super_secret_key_change_me_in_production_1234567890")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database (PostgreSQL default, fallback to SQLite)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./sql_app.db")
    
    # Redis (Optional, default to None)
    REDIS_URL: str = os.getenv("REDIS_URL", "")
    
    # Uploads
    UPLOAD_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads")
    
    # CORS Origins
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:3000",
    ]
    
    # Third Party APIs
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure upload directory exists
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
