import os
import shutil
import uuid
from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from pymongo.database import Database
from pypdf import PdfReader

from backend.app.db.session import get_db
from backend.app.schemas.document import DocumentOut
from backend.app.api.deps import get_current_user
from backend.app.core.config import settings

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_pdf_content(file_path: str) -> str:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error parsing PDF: {str(e)}",
        )


def _doc_out(doc: dict) -> dict:
    return {
        "id": doc["_id"],
        "user_id": doc["user_id"],
        "filename": doc["filename"],
        "file_path": doc["file_path"],
        "content": doc["content"],
        "file_size": doc["file_size"],
        "created_at": doc["created_at"],
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=List[DocumentOut])
def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    saved_docs = []
    user_id = current_user["_id"]
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)

    for file in files:
        name = file.filename.lower()
        if not (name.endswith(".txt") or name.endswith(".pdf")):
            continue

        file_path = os.path.join(user_upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_size = os.path.getsize(file_path)

        if name.endswith(".pdf"):
            content = parse_pdf_content(file_path)
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                if os.path.exists(file_path):
                    os.remove(file_path)
                continue

        doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "filename": file.filename,
            "file_path": file_path,
            "content": content,
            "file_size": file_size,
            "created_at": datetime.utcnow(),
        }
        db["documents"].insert_one(doc)
        saved_docs.append(_doc_out(doc))

    return saved_docs


@router.get("/", response_model=List[DocumentOut])
def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    docs = db["documents"].find({"user_id": current_user["_id"]})
    return [_doc_out(d) for d in docs]


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    doc = db["documents"].find_one(
        {"_id": document_id, "user_id": current_user["_id"]}
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    if os.path.exists(doc["file_path"]):
        try:
            os.remove(doc["file_path"])
        except Exception:
            pass

    db["documents"].delete_one({"_id": document_id})
    return None


@router.post("/preload-corpus", response_model=List[DocumentOut])
def preload_corpus(
    corpus_key: str,
    current_user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    # Locate workspace root containing research_documents/
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = current_dir
    while workspace_root and workspace_root != "/":
        if os.path.exists(os.path.join(workspace_root, "research_documents")):
            break
        workspace_root = os.path.dirname(workspace_root)
    else:
        workspace_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        )

    corpus_mapping = {
        "primary": os.path.join(workspace_root, "research_documents"),
        "pdf_papers": os.path.join(workspace_root, "research_documents", "pdf_papers"),
        "semantic_demo": os.path.join(workspace_root, "research_documents", "semantic_demo"),
    }

    target_dir = corpus_mapping.get(corpus_key)
    if not target_dir or not os.path.exists(target_dir):
        if corpus_key == "primary":
            from backend.create_corpus import generate_text_corpus
            generate_text_corpus(os.path.join(workspace_root, "research_documents"))
            target_dir = os.path.join(workspace_root, "research_documents")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Requested corpus directory does not exist: {corpus_key}",
            )

    saved_docs = []
    user_id = current_user["_id"]
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)

    for fname in sorted(os.listdir(target_dir)):
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if not (fname.lower().endswith(".txt") or fname.lower().endswith(".pdf")):
            continue

        # Skip if already loaded for this user
        existing = db["documents"].find_one({"user_id": user_id, "filename": fname})
        if existing:
            saved_docs.append(_doc_out(existing))
            continue

        dest_path = os.path.join(user_upload_dir, fname)
        shutil.copy(fpath, dest_path)
        file_size = os.path.getsize(dest_path)

        if fname.lower().endswith(".pdf"):
            content = parse_pdf_content(dest_path)
        else:
            try:
                with open(dest_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                continue

        doc = {
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "filename": fname,
            "file_path": dest_path,
            "content": content,
            "file_size": file_size,
            "created_at": datetime.utcnow(),
        }
        db["documents"].insert_one(doc)
        saved_docs.append(_doc_out(doc))

    return saved_docs
