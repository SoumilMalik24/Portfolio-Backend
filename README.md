# Portfolio Backend

Backend API for Soumil Malik's portfolio website, featuring a RAG-based AI assistant, contact form submission, and deployment-ready architecture.

## Features

- **RAG-Based AI Assistant**: Uses LangChain with FAISS vector store and Hugging Face embeddings for intelligent Q&A about Soumil's profile.
- **Contact Form**: Secure email submission via SMTP (Gmail supported).
- **Production Ready**: Built with FastAPI, CORS middleware, and environment variable management.
- **Auto-Index Rebuilding**: Automatically rebuilds the FAISS index when `portfolio_data.txt` changes.

## Tech Stack

- **Framework**: FastAPI
- **AI/ML**: LangChain, FAISS, HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **Database**: FAISS (in-memory vector store)
- **Deployment**: Docker, Render (tested)
- **Environment**: Python 3.10+

## Setup

1.  **Clone the repository**
    ```bash
    git clone <repository-url>
    cd portfolio-backend
    ```

2.  **Create a virtual environment**
    ```bash
    python -m venv .venv
    source .venv/Scripts/activate  # Windows
    # source .venv/bin/activate  # Linux/Mac
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Configuration**
    Create a `.env` file in the root directory:
    ```env
    HUGGINGFACE_API_TOKEN=your_huggingface_token
    EMAIL_HOST=smtp.gmail.com
    EMAIL_PORT=587
    EMAIL_USER=your_email@gmail.com
    EMAIL_PASS=your_app_password
    FRONTEND_URL=http://localhost:5173
    ```

5.  **Run the Server**
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://localhost:8000`.

## API Endpoints

### 1. Health Check
```http
GET /api/
```
**Response:**
```json
{
  "status": "API running 🚀"
}
```

### 2. Chat with AI Assistant
```http
POST /api/chat
```
**Request Body:**
```json
{
  "message": "What technologies do you specialize in?"
}
```

**Response:**
```json
{
  "reply": "I specialize in Generative AI and MLOps, with hands-on experience in RAG pipelines, LLM orchestration, vector databases, and cloud deployment."
}
```

### 3. Contact Form Submission
```http
POST /api/contact
```
**Request Body:**
```json
{
  "email": "[EMAIL_ADDRESS]",
  "subject": "Project Inquiry",
  "message": "I'd like to discuss a potential collaboration."
}
```

**Response:**
```json
{
  "status": "Email sent successfully"
}
```

## Data Management

The RAG system uses `portfolio_data.txt` as its knowledge base.
- **Auto-Indexing**: The first time the server starts, or whenever `portfolio_data.txt` is modified, the FAISS index is automatically rebuilt.
- **Persistence**: The index is saved locally in the `faiss_index` directory.

## Deployment

To deploy on Render:
1.  Push your code to GitHub.
2.  Create a new "Web Service" on Render.
3.  Connect your GitHub repository.
4.  Set the **Build Command**: `pip install -r requirements.txt`
5.  Set the **Start Command**: `uvicorn main:app --host $PORT --workers 1`
6.  Add your environment variables (HUGGINGFACE_API_TOKEN, EMAIL_*, FRONTEND_URL).
7.  Deploy!
