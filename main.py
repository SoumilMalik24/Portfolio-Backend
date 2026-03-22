import os
import smtplib
from email.mime.text import MIMEText

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from dotenv import load_dotenv

from rag import RAGEngine

load_dotenv()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()

# ------------------ MODELS ------------------

class ChatRequest(BaseModel):
    message: str

class ContactRequest(BaseModel):
    email: EmailStr
    subject: str
    message: str

# ------------------ ROUTES ------------------

@app.get("/")
def root():
    return {"status": "API running 🚀"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    try:
        reply = rag.query(req.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contact")
def contact(req: ContactRequest):
    try:
        html_content = f"""
        <h2>New Portfolio Contact</h2>
        <p><strong>Email:</strong> {req.email}</p>
        <p><strong>Subject:</strong> {req.subject}</p>
        <p><strong>Message:</strong><br>{req.message}</p>
        """

        msg = MIMEText(html_content, "html")
        msg["Subject"] = f"Portfolio Contact: {req.subject}"
        msg["From"] = os.getenv("EMAIL_USER")
        msg["To"] = os.getenv("EMAIL_USER")

        with smtplib.SMTP(os.getenv("EMAIL_HOST"), int(os.getenv("EMAIL_PORT"))) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(msg)

        return {"status": "Email sent successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))