import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Traverse upwards from this file to find and load the .env file
_current_dir = os.path.dirname(os.path.abspath(__file__))
while _current_dir and _current_dir != "/":
    _env_path = os.path.join(_current_dir, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
        break
    _current_dir = os.path.dirname(_current_dir)
else:
    load_dotenv()


def _parse_cors_origins() -> List[str]:
    """
    Build the CORS allowed-origins list.
    In production, set FRONTEND_URL to your deployed frontend origin.
    Multiple origins can be provided as a comma-separated string.
    """
    defaults = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
    ]
    env_val = os.getenv("FRONTEND_URL", "")
    if env_val:
        extra = [o.strip() for o in env_val.split(",") if o.strip()]
        seen = set(defaults)
        for origin in extra:
            if origin not in seen:
                defaults.append(origin)
                seen.add(origin)
    return defaults


class Settings(BaseSettings):
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "NLP Research Analyzer"

    # ── Security ──────────────────────────────────────────────────────────────
    # IMPORTANT: Override SECRET_KEY in production with a long random string.
    # Generate one: python -c "import secrets; print(secrets.token_hex(32))"
    SECRET_KEY: str = os.getenv(
        "SECRET_KEY",
        "insecure_default_please_change_in_production_1234567890abcdef"
    )
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # ── MongoDB ───────────────────────────────────────────────────────────────
    # Local:  mongodb://localhost:27017
    # Atlas:  mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority
    MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    MONGODB_DB_NAME: str = os.getenv("MONGODB_DB_NAME", "nlp_research")

    # ── Redis (Optional) ──────────────────────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "")

    # ── File Uploads ──────────────────────────────────────────────────────────
    UPLOAD_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "uploads"
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    BACKEND_CORS_ORIGINS: List[str] = _parse_cors_origins()

    # ── Third-Party APIs ──────────────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    class Config:
        case_sensitive = True


settings = Settings()

# Ensure the upload directory exists at startup
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
