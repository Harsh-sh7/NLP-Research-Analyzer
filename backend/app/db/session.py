from pymongo import MongoClient, ASCENDING
from pymongo.database import Database
from backend.app.core.config import settings

# ── Singleton MongoClient ─────────────────────────────────────────────────────
# A single MongoClient instance is reused across the entire app lifetime.
# PyMongo's MongoClient manages its own internal connection pool, so one
# instance per process is correct and thread-safe.
_client: MongoClient | None = None


def get_client() -> MongoClient:
    """Return the shared MongoClient, creating it on first call."""
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI)
    return _client


def get_db() -> Database:
    """
    FastAPI dependency — yields the MongoDB database object.
    Usage in a route: db: Database = Depends(get_db)
    """
    return get_client()[settings.MONGODB_DB_NAME]


def create_indexes() -> None:
    """
    Create all necessary MongoDB indexes at application startup.
    Safe to call multiple times (createIndex is idempotent).
    """
    db = get_db()

    # users: unique email lookup + username lookup
    db["users"].create_index([("email", ASCENDING)], unique=True)
    db["users"].create_index([("username", ASCENDING)])

    # documents: filter by owner, check duplicate filenames per user
    db["documents"].create_index([("user_id", ASCENDING)])
    db["documents"].create_index([("user_id", ASCENDING), ("filename", ASCENDING)])

    # analysis_jobs: list by owner, sort by creation date
    db["analysis_jobs"].create_index([("user_id", ASCENDING)])
    db["analysis_jobs"].create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])

    # research_jobs: list by owner, sort by creation date
    db["research_jobs"].create_index([("user_id", ASCENDING)])
    db["research_jobs"].create_index([("user_id", ASCENDING), ("created_at", ASCENDING)])

    # reports: list by owner; unique mapping to research job
    db["reports"].create_index([("user_id", ASCENDING)])
    db["reports"].create_index([("job_id", ASCENDING)], unique=True)
