import os

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient

from utils import get_file_hash, load_metadata, save_metadata

VECTOR_PATH = "faiss_index"
DATA_FILE = "portfolio_data.txt"


class RAGEngine:
    def __init__(self):
        # Embeddings (Hugging Face Inference API - Free & 0 MB RAM)
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="sentence-transformers/all-MiniLM-L6-v2",
            task="feature-extraction",
            huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN")
        )

        # Initialize vector DB
        self.vectorstore = self._initialize_vectorstore()

        # LLM (Hugging Face Inference API - free)
        self.client = InferenceClient(api_key=os.getenv("HUGGINGFACE_API_TOKEN"))

    def _initialize_vectorstore(self):
        if not os.path.exists(DATA_FILE):
            raise FileNotFoundError("portfolio_data.txt not found")

        current_hash = get_file_hash(DATA_FILE)
        metadata = load_metadata()
        stored_hash = metadata.get("file_hash")

        # Rebuild if:
        # - index doesn't exist
        # - OR file changed
        if not os.path.exists(VECTOR_PATH) or current_hash != stored_hash:
            print("Rebuilding FAISS index...")

            with open(DATA_FILE, "r", encoding="utf-8") as f:
                text = f.read().strip()

            if not text:
                raise ValueError("portfolio_data.txt is EMPTY")

            splitter = CharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

            chunks = splitter.split_text(text)

            if not chunks:
                raise ValueError("No chunks created from text")

            docs = [Document(page_content=chunk) for chunk in chunks]

            # Create FAISS index
            vectorstore = FAISS.from_documents(docs, self.embeddings)

            # Save locally
            vectorstore.save_local(VECTOR_PATH)

            # Save hash
            save_metadata({"file_hash": current_hash})

            print("FAISS index rebuilt successfully")

            return vectorstore

        # Load existing index
        print("Loading existing FAISS index...")
        return FAISS.load_local(
            VECTOR_PATH,
            self.embeddings,
            allow_dangerous_deserialization=True
        )

    def query(self, question: str) -> str:
        if not question.strip():
            return "Please ask a valid question."

        # Retrieve top matches
        docs = self.vectorstore.similarity_search(question, k=3)

        if not docs:
            return "I don't have that information."

        context = "\n".join([doc.page_content for doc in docs])

        # Strict system prompt
        system_prompt = f"""
You are Soumil's helpful AI assistant.

STRICT RULES:
- Keep answers EXTREMELY short and concise (1 to 2 sentences max).
- Provide only the specific information requested, DO NOT summarize the whole context.
- Answer ONLY using the provided context.
- If answer is not in context, say: "I don't have that information."
- Do NOT answer general knowledge questions.

Context:
{context}
"""

        try:
            response = self.client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=150
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print("LLM Error:", str(e))
            return "Something went wrong while generating response."