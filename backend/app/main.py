import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.db.session import engine, Base
from backend.app.api.endpoints import auth, documents, nlp, research
from sqlalchemy import text

# Create database tables automatically
Base.metadata.create_all(bind=engine)

# SQLite backward compatibility column migration
try:
    with engine.begin() as conn:
        result = conn.execute(text("PRAGMA table_info(users)"))
        columns = [row[1] for row in result.fetchall()]
        if "username" not in columns:
            conn.execute(text("ALTER TABLE users ADD COLUMN username VARCHAR"))
except Exception as e:
    print(f"Database migration note: {e}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS configurations
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Include routers
app.include_router(auth.router, prefix=f"{settings.API_V1_STR}/auth", tags=["auth"])
app.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["documents"])
app.include_router(nlp.router, prefix=f"{settings.API_V1_STR}/nlp", tags=["nlp"])
app.include_router(research.router, prefix=f"{settings.API_V1_STR}/research", tags=["research"])

@app.get("/")
def read_root():
    return {"status": "ok", "project": settings.PROJECT_NAME}
