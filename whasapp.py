import asyncpg
from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
import jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here")  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# WhatsApp Business API configuration
WHATSAPP_API_URL = "https://graph.facebook.com/v20.0"
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")

# PostgreSQL configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT", 5432)
}

# Pydantic models
class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Database functions
async def get_user_from_db(username: str):
    """Get user from database by username."""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        user = await conn.fetchrow(
            "SELECT username, password_hash, is_active FROM users WHERE username = $1",
            username
        )
        await conn.close()
        return user
    except Exception as e:
        print(f"Database error: {str(e)}")
        return None

async def get_contacts():
    """Fetch all contact numbers from the database."""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        contacts = await conn.fetch("SELECT phone_number FROM contacts WHERE active = TRUE")
        await conn.close()
        return [contact["phone_number"] for contact in contacts]
    except Exception as e:
        print(f"Database error: {str(e)}")
        return []

# Authentication functions
def verify_password(plain_password, hashed_password):
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Hash a password."""
    return pwd_context.hash(password)

async def authenticate_user(username: str, password: str):
    """Authenticate a user."""
    user = await get_user_from_db(username)
    if not user:
        return False
    if not user["is_active"]:
        return False
    if not verify_password(password, user["password_hash"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except jwt.PyJWTError:
        raise credentials_exception
    
    user = await get_user_from_db(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: dict = Depends(get_current_user)):
    """Get current active user."""
    if not current_user["is_active"]:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# WhatsApp functions
async def send_whatsapp_message(to_number: str, text: str, media_url: str = None):
    """Send WhatsApp message to a single contact using WhatsApp Business API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number,
        "type": "text",
        "text": {"body": text}
    }
    
    if media_url:
        payload["type"] = "image"
        payload["image"] = {
            "link": media_url,
            "caption": text  # Add caption to image
        }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                f"{WHATSAPP_API_URL}/{WHATSAPP_PHONE_NUMBER_ID}/messages",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            return {"status": "success", "message_id": response.json().get("messages", [{}])[0].get("id")}
        except httpx.HTTPStatusError as e:
            return {"status": "error", "error": f"HTTP error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Authentication endpoints
@app.post("/auth/login", response_model=Token)
async def login(user_credentials: UserLogin):
    """Login endpoint to get JWT token."""
    user = await authenticate_user(user_credentials.username, user_credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me")
async def read_users_me(current_user: dict = Depends(get_current_active_user)):
    """Get current user information."""
    return {"username": current_user["username"], "is_active": current_user["is_active"]}

# Protected WhatsApp endpoint
@app.post("/send-whatsapp")
async def send_whatsapp(
    text: str = Form(...),
    image: UploadFile = File(None),
    current_user: dict = Depends(get_current_active_user)
):
    """Protected endpoint to receive text and optional image and send to all contacts."""
    try:
        # Save uploaded image if provided
        media_url = None
        if image:
            # In production, upload to a cloud storage service (e.g., AWS S3) and get a public URL
            # This is a placeholder for where you'd upload the image
            media_url = "https://your-image-hosting-service.com/uploaded-image.jpg"
        
        # Get contacts from database
        contacts = await get_contacts()
        
        if not contacts:
            return JSONResponse(
                status_code=400,
                content={"message": "No contacts found in database"}
            )
        
        # Send messages to all contacts
        results = []
        success_count = 0
        failed_count = 0
        
        for contact in contacts:
            result = await send_whatsapp_message(contact, text, media_url)
            results.append({"contact": contact, **result})
            if result["status"] == "success":
                success_count += 1
            else:
                failed_count += 1
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Messages processed",
                "success_count": success_count,
                "failed_count": failed_count,
                "results": results,
                "sent_by": current_user["username"]
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing request: {str(e)}"}
        )

# Utility endpoint to create users (remove in production or add admin protection)
@app.post("/auth/create-user")
async def create_user(username: str = Form(...), password: str = Form(...)):
    """Create a new user (for testing purposes)."""
    try:
        conn = await asyncpg.connect(**DB_CONFIG)
        
        # Check if user already exists
        existing_user = await conn.fetchrow(
            "SELECT username FROM users WHERE username = $1", username
        )
        if existing_user:
            await conn.close()
            raise HTTPException(
                status_code=400,
                detail="Username already registered"
            )
        
        # Hash password and create user
        hashed_password = get_password_hash(password)
        await conn.execute(
            "INSERT INTO users (username, password_hash, is_active) VALUES ($1, $2, $3)",
            username, hashed_password, True
        )
        await conn.close()
        
        return {"message": "User created successfully"}
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error creating user: {str(e)}"}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
