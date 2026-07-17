from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routers import chat
from app.core.database import engine, Base

# Auto-create the database tables based on our models
Base.metadata.create_all(bind=engine)

app = FastAPI(title="InsureHub AI Bot")

# Allow the React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the chat route
app.include_router(chat.router)

@app.get("/")
def root():
    return {"status": "InsureHub AI Backend is active!"}