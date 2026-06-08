import random
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pymongo.database import Database
from pymongo.errors import DuplicateKeyError

from backend.app.db.session import get_db
from backend.app.schemas.user import UserCreate, UserOut, Token
from backend.app.core.security import get_password_hash, verify_password, create_access_token
from backend.app.api.deps import get_current_user

router = APIRouter()


@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, db: Database = Depends(get_db)):
    # Check email uniqueness
    if db["users"].find_one({"email": user_in.email}):
        raise HTTPException(
            status_code=400,
            detail="A user with this email address already exists in the system.",
        )

    # Resolve username
    base_username = (
        user_in.username.strip()
        if user_in.username and user_in.username.strip()
        else user_in.email.split("@")[0]
    )

    # Ensure username uniqueness
    username = base_username
    for _ in range(100):
        if not db["users"].find_one({"username": username}):
            break
        username = f"{base_username}{random.randint(10, 99)}"
    else:
        username = f"{base_username}_{uuid.uuid4().hex[:4]}"

    user_doc = {
        "_id": str(uuid.uuid4()),
        "email": user_in.email,
        "username": username,
        "password_hash": get_password_hash(user_in.password),
        "created_at": datetime.utcnow(),
    }

    try:
        db["users"].insert_one(user_doc)
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Email already registered.")

    return _user_out(user_doc)


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Database = Depends(get_db)):
    user = db["users"].find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password",
        )
    access_token = create_access_token(subject=user["_id"])
    return {"access_token": access_token, "token_type": "bearer"}


@router.get("/me", response_model=UserOut)
def read_user_me(current_user: dict = Depends(get_current_user)):
    return _user_out(current_user)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_out(doc: dict) -> dict:
    """Map a MongoDB user document to the UserOut response shape."""
    return {
        "id": doc["_id"],
        "email": doc["email"],
        "username": doc.get("username"),
        "created_at": doc["created_at"],
    }
