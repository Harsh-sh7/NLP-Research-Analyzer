import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from backend.app.db.session import Base

def generate_uuid():
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, index=True, nullable=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    nlp_jobs = relationship("AnalysisJob", back_populates="user", cascade="all, delete-orphan")
    research_jobs = relationship("ResearchJob", back_populates="user", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="documents")

class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    document_ids = Column(JSON, nullable=False)  # List of document UUIDs analyzed
    vectorization_mode = Column(String, nullable=False)  # TF-IDF vs SBERT
    parameters = Column(JSON, nullable=False)  # Preprocessing and model parameters
    results = Column(JSON, nullable=False)  # Heatmap, PCA coords, cluster keywords, summaries
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="nlp_jobs")

class ResearchJob(Base):
    __tablename__ = "research_jobs"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    query = Column(String, nullable=False)
    status = Column(String, default="pending", nullable=False)
    task_list = Column(JSON, default=list, nullable=False)
    scraped_urls = Column(JSON, default=list, nullable=False)
    citations = Column(JSON, default=dict, nullable=False)
    report_draft = Column(Text, default="", nullable=False)
    revision_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="research_jobs")
    report = relationship("Report", back_populates="job", uselist=False, cascade="all, delete-orphan")

class Report(Base):
    __tablename__ = "reports"

    id = Column(String, primary_key=True, default=generate_uuid, index=True)
    job_id = Column(String, ForeignKey("research_jobs.id", ondelete="CASCADE"), unique=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    metrics = Column(JSON, nullable=False)
    citations = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="reports")
    job = relationship("ResearchJob", back_populates="report")
