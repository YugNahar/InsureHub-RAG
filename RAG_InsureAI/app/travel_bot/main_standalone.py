"""
Standalone entrypoint for running the travel-bot backend on its own,
separate from the main InsureAI RAG app — mirrors Liza's original
insurehub-ai-backend/app/main.py. Run from RAG_InsureAI/app/ so the
travel_bot package resolves:
    uvicorn travel_bot.main_standalone:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from travel_bot.routers import chat
from travel_bot.core.database import engine, Base

Base.metadata.create_all(bind=engine)

app = FastAPI(title="InsureHub Travel Bot (standalone)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)

@app.get("/")
def root():
    return {"status": "InsureHub Travel Bot standalone backend is active!"}
