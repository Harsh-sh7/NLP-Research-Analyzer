"""
MongoDB document shape definitions.

These are plain typed dicts used for documentation and type-checking only.
PyMongo stores and retrieves plain Python dicts — no ORM magic needed.

Each collection stores documents with the following fields:

users:
  _id         : str  (UUID string)
  email       : str  (unique)
  username    : str
  password_hash: str
  created_at  : datetime

documents:
  _id         : str  (UUID string)
  user_id     : str  (ref → users._id)
  filename    : str
  file_path   : str
  content     : str  (extracted text)
  file_size   : int  (bytes)
  created_at  : datetime

analysis_jobs:
  _id                : str  (UUID string)
  user_id            : str  (ref → users._id)
  document_ids       : list[str]
  vectorization_mode : str
  parameters         : dict
  results            : dict
  created_at         : datetime

research_jobs:
  _id            : str  (UUID string)
  user_id        : str  (ref → users._id)
  query          : str
  status         : str  ("pending" | "running" | "completed" | "failed")
  task_list      : list
  scraped_urls   : list
  citations      : dict
  report_draft   : str
  revision_count : int
  created_at     : datetime

reports:
  _id        : str  (UUID string)
  job_id     : str  (ref → research_jobs._id, unique)
  user_id    : str  (ref → users._id)
  title      : str
  content    : str
  metrics    : dict
  citations  : dict
  created_at : datetime
"""

import uuid
from datetime import datetime


def new_id() -> str:
    """Generate a new UUID4 string to use as a MongoDB document _id."""
    return str(uuid.uuid4())


def now() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()
