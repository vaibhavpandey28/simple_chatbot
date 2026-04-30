from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
from core.logger import get_logger
from service.llm import get_response

app = FastAPI()
logger = get_logger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class ChatRequest(BaseModel):
    session_id: str
    user_input: str

class ChatResponse(BaseModel):
    reply: str

@app.get("/")
def root():
    logger.info("UI endpoint hit")
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    logger.info("Health check endpoint hit")
    return {"message": "Chatbot API is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest):
    logger.info("Received chat request for session_id=%s", chat_request.session_id)
    reply = get_response(chat_request.session_id, chat_request.user_input)
    logger.info("Returning response for session_id=%s", chat_request.session_id)
    return ChatResponse(reply=reply)
