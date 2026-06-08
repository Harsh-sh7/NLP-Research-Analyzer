import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.db.session import create_indexes
from backend.app.api.endpoints import auth, documents, nlp, research

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(o) for o in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,      prefix=f"{settings.API_V1_STR}/auth",      tags=["auth"])
app.include_router(documents.router, prefix=f"{settings.API_V1_STR}/documents", tags=["documents"])
app.include_router(nlp.router,       prefix=f"{settings.API_V1_STR}/nlp",       tags=["nlp"])
app.include_router(research.router,  prefix=f"{settings.API_V1_STR}/research",  tags=["research"])


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    """Create MongoDB indexes on application start (idempotent)."""
    create_indexes()


@app.get("/")
def read_root():
    return {"status": "ok", "project": settings.PROJECT_NAME}
