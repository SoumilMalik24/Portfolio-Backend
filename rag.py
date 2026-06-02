import os
import requests

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils import get_file_hash, load_metadata, save_metadata

VECTOR_PATH = "faiss_index"
DATA_FILE   = "portfolio_data.txt"
MODEL       = "nvidia/nemotron-3-nano-30b-a3b:free"
OR_URL      = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are an AI assistant embedded in Soumil Malik's portfolio website.
Your job is to help visitors learn about Soumil — his skills, projects, experience, education, and background.

STRICT RULES:
- Answer ONLY using the retrieved context provided below.
- Keep answers concise and professional (1-2 sentences max).
- Do NOT answer general knowledge questions unrelated to Soumil.
- If the answer is not in the context, respond with: "I don't have that information."
- You may guide visitors on how to contact Soumil if asked.

Context:
{context}
"""


class RAGEngine:
    def __init__(self):
        # Local embeddings (no API key required, runs locally via sentence-transformers)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = self._initialize_vectorstore()
        self.or_api_key  = os.getenv("OPENROUTER_API_KEY")

    # ── Vector store ──────────────────────────────────────────────────────────

    def _initialize_vectorstore(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError(f"{DATA_FILE} not found")

        current_hash = get_file_hash(DATA_FILE)
        metadata     = load_metadata()
        stored_hash  = metadata.get("file_hash")

        if not os.path.exists(VECTOR_PATH) or current_hash != stored_hash:
            print("🔄 Rebuilding FAISS index...")
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                raise ValueError(f"{DATA_FILE} is empty")

            splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks   = splitter.split_text(text)
            docs     = [Document(page_content=c) for c in chunks]

            vectorstore = FAISS.from_documents(docs, self.embeddings)
            vectorstore.save_local(VECTOR_PATH)
            save_metadata({"file_hash": current_hash})
            print("✅ FAISS index rebuilt")
            return vectorstore

        print("✅ Loading existing FAISS index...")
        return FAISS.load_local(
            VECTOR_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, history: list = None) -> str:
        """
        Retrieve relevant context and call OpenRouter with full conversation history.

        Args:
            question: Latest user question (used for retrieval).
            history:  List of ChatMessage objects (role + content).
        """
        if not question.strip():
            return "Please ask a valid question."

        # 1. Retrieve top-k matching chunks
        docs    = self.vectorstore.similarity_search(question, k=4)
        context = "\n\n".join(d.page_content for d in docs)

        # 2. Build system message with injected context
        system_msg = {"role": "system", "content": SYSTEM_PROMPT.format(context=context)}

        # 3. Build conversation messages
        #    We include prior history so the model can follow the thread
        prior = []
        if history:
            for msg in history:
                # Skip the very last user message — we'll add it fresh below
                if msg == history[-1] and msg.role == "user":
                    continue
                prior.append({"role": msg.role, "content": msg.content})

        messages = [system_msg] + prior + [{"role": "user", "content": question}]

        # 4. Call OpenRouter
        try:
            response = requests.post(
                url=OR_URL,
                headers={
                    "Authorization": f"Bearer {self.or_api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://soumilmalik.dev",
                    "X-Title": "Soumil Malik Portfolio",
                },
                json={
                    "model": MODEL,
                    "messages": messages,
                    "temperature": 0.4,
                    "max_tokens": 300,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()

        except requests.exceptions.Timeout:
            return "The AI is taking too long to respond. Please try again."
        except Exception as e:
            print("OpenRouter error:", e)
            return "Something went wrong while generating a response. Please try again."