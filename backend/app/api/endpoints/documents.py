import os
import shutil
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from pypdf import PdfReader

from backend.app.db.session import get_db
from backend.app.db.models import Document, User
from backend.app.schemas.document import DocumentOut
from backend.app.api.deps import get_current_user
from backend.app.core.config import settings

router = APIRouter()

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
            detail=f"Error parsing PDF: {str(e)}"
        )

@router.post("/upload", response_model=List[DocumentOut])
def upload_documents(
    files: List[UploadFile] = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    saved_docs = []
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, current_user.id)
    os.makedirs(user_upload_dir, exist_ok=True)

    for file in files:
        if not (file.filename.lower().endswith(".txt") or file.filename.lower().endswith(".pdf")):
            continue
            
        file_path = os.path.join(user_upload_dir, file.filename)
        
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        file_size = os.path.getsize(file_path)
        
        # Extract content
        if file.filename.lower().endswith(".pdf"):
            content = parse_pdf_content(file_path)
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                continue
                
        # Save to database
        db_doc = Document(
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            content=content,
            file_size=file_size
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        saved_docs.append(db_doc)
        
    return saved_docs

@router.get("/", response_model=List[DocumentOut])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Document).filter(Document.user_id == current_user.id).all()

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(
        Document.id == document_id, 
        Document.user_id == current_user.id
    ).first()
    
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
        
    # Remove from disk if exists
    if os.path.exists(doc.file_path):
        try:
            os.remove(doc.file_path)
        except Exception:
            pass
            
    db.delete(doc)
    db.commit()
    return None

@router.post("/preload-corpus", response_model=List[DocumentOut])
def preload_corpus(
    corpus_key: str,  # "primary", "pdf_papers", "semantic_demo"
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Traverse upwards to find workspace root containing research_documents folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_root = current_dir
    while workspace_root and workspace_root != "/":
        if os.path.exists(os.path.join(workspace_root, "research_documents")):
            break
        workspace_root = os.path.dirname(workspace_root)
    else:
        workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
    corpus_mapping = {
        "primary": os.path.join(workspace_root, "research_documents"),
        "pdf_papers": os.path.join(workspace_root, "research_documents", "pdf_papers"),
        "semantic_demo": os.path.join(workspace_root, "research_documents", "semantic_demo")
    }
    
    target_dir = corpus_mapping.get(corpus_key)
    if not target_dir or not os.path.exists(target_dir):
        # Create standard text corpus if primary and missing
        if corpus_key == "primary":
            from create_corpus import generate_text_corpus
            generate_text_corpus(os.path.join(workspace_root, "research_documents"))
            target_dir = os.path.join(workspace_root, "research_documents")
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Requested corpus directory does not exist: {corpus_key}"
            )
            
    saved_docs = []
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, current_user.id)
    os.makedirs(user_upload_dir, exist_ok=True)
    
    for fname in sorted(os.listdir(target_dir)):
        fpath = os.path.join(target_dir, fname)
        if not os.path.isfile(fpath):
            continue
            
        if not (fname.lower().endswith(".txt") or fname.lower().endswith(".pdf")):
            continue
            
        # Check if already loaded
        existing = db.query(Document).filter(
            Document.user_id == current_user.id,
            Document.filename == fname
        ).first()
        if existing:
            saved_docs.append(existing)
            continue
            
        dest_path = os.path.join(user_upload_dir, fname)
        shutil.copy(fpath, dest_path)
        file_size = os.path.getsize(dest_path)
        
        # Extract content
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
                
        # Save to database
        db_doc = Document(
            user_id=current_user.id,
            filename=fname,
            file_path=dest_path,
            content=content,
            file_size=file_size
        )
        db.add(db_doc)
        db.commit()
        db.refresh(db_doc)
        saved_docs.append(db_doc)
        
    return saved_docs
