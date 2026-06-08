from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from backend.app.db.session import get_db
from backend.app.db.models import User
from backend.app.schemas.user import UserCreate, UserOut, Token
from backend.app.core.security import get_password_hash, verify_password, create_access_token
from backend.app.api.deps import get_current_user

router = APIRouter()

@router.post("/signup", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def signup(user_in: UserCreate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_in.email).first()
    if user:
        raise HTTPException(
            status_code=400,
            detail="A user with this email address already exists in the system.",
        )
    
    # Username processing
    base_username = user_in.username.strip() if user_in.username and user_in.username.strip() else user_in.email.split("@")[0]
    
    # Ensure uniqueness in the database
    import random
    username = base_username
    for _ in range(100):  # Try up to 100 times to find a unique suffix if collisions occur
        existing = db.query(User).filter(User.username == username).first()
        if not existing:
            break
        username = f"{base_username}{random.randint(10, 99)}"
    else:
        # Fallback to appending a UUID fragment if 100 random attempts somehow collide
        import uuid
        username = f"{base_username}_{uuid.uuid4().hex[:4]}"

    hashed_pwd = get_password_hash(user_in.password)
    db_user = User(email=user_in.email, username=username, password_hash=hashed_pwd)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(subject=user.id)
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me", response_model=UserOut)
def read_user_me(current_user: User = Depends(get_current_user)):
    return current_user
