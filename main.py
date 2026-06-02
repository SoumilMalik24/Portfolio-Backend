import os
import requests
import smtplib
from email.mime.text import MIMEText

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from rag import RAGEngine

load_dotenv()

app = FastAPI(title="Soumil Malik — Portfolio API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── RAG engine (loads on startup) ─────────────────────────────────────────────
rag = RAGEngine()

# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str        # "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: list[ChatMessage]   # full conversation history from client

class ContactRequest(BaseModel):
    email: EmailStr
    subject: str
    message: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Portfolio API is running 🚀"}


@app.head("/health")
def health():
    """Health check endpoint — use this to verify the server is alive."""
    return {"status": "ok", "service": "portfolio-api"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    """
    RAG chatbot endpoint.
    Accepts full conversation history and returns the assistant's next reply.
    """
    try:
        if not req.messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        # Get the latest user message for retrieval
        user_messages = [m for m in req.messages if m.role == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found")

        latest_query = user_messages[-1].content
        reply = rag.query(latest_query, req.messages)
        return {"reply": reply}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contact")
def contact(req: ContactRequest):
    """
    Contact form endpoint.
    Sends an email to Soumil via Gmail SMTP using the app password.
    """
    try:
        html_content = f"""
        <h2 style="color:#c95f2a;">New Portfolio Contact</h2>
        <table style="font-family:sans-serif;font-size:14px;">
          <tr><td><strong>From:</strong></td><td>{req.email}</td></tr>
          <tr><td><strong>Subject:</strong></td><td>{req.subject}</td></tr>
        </table>
        <hr/>
        <p style="font-family:sans-serif;font-size:14px;line-height:1.7;">
          {req.message.replace(chr(10), '<br>')}
        </p>
        """

        msg = MIMEText(html_content, "html")
        msg["Subject"] = f"[Portfolio] {req.subject}"
        msg["From"]    = os.getenv("EMAIL_USER")
        msg["To"]      = os.getenv("EMAIL_USER")
        msg["Reply-To"] = req.email   # so you can reply directly

        with smtplib.SMTP(os.getenv("EMAIL_HOST"), int(os.getenv("EMAIL_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(msg)

        return {"status": "Message sent successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))