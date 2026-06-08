# NLP Research Analyzer

A full-stack AI-powered research platform that combines **Classical NLP Analysis** with an **Agentic AI Research Swarm** — built with a React + Vite frontend and a FastAPI backend.

Upload documents, run deep linguistic analysis, or deploy an autonomous multi-agent swarm to research any topic from the live web — all from a sleek, dark-mode UI.

---

## ✨ Features

### 📄 NLP Workspace
Analyze your own private research documents (PDF or TXT) with a full classical NLP pipeline:

- **Dual-Mode Vectorization** — Switch between TF-IDF (lexical) and Semantic Embeddings (SBERT `all-MiniLM-L6-v2`)
- **Similarity Heatmap** — Pairwise cosine similarity across all uploaded documents
- **K-Means Clustering** — Automatically groups documents into coherent thematic clusters with Silhouette Score optimization
- **LDA Topic Modeling** — Surfaces latent hidden topics across the entire corpus
- **PCA Visualization** — 2D scatter plot of document embeddings in vector space
- **Cluster Results** — Per-cluster keyword extraction and extractive TextRank summaries
- **Interactive Charts** — All visualizations built with Recharts, fully interactive

### 🤖 Agentic Research Swarm
Deploy an autonomous multi-node LangGraph agent to research any topic from the live web:

- **Query Decomposition** — Breaks a broad topic into 3–5 targeted sub-questions
- **Live Web Search** — Real-time web search via the Tavily API
- **Deep Scraping + RAG** — Visits top URLs, chunks and embeds text via FAISS, performs retrieval-augmented generation
- **Structured Report Synthesis** — Generates a professional Markdown report with cited sources
- **Automated Quality Review** — A Reviewer node validates drafts and triggers rework cycles if quality thresholds aren't met
- **Agent Console** — Live terminal-style feed showing each node as it executes
- **Knowledge Graph** — Visual node graph of the agent pipeline (ReactFlow)
- **Reports Archive** — All generated reports are saved and downloadable

### 🔐 Auth & Persistence
- JWT-based authentication (signup / login)
- Username personalization across the app
- Document history and research job history per user
- SQLite (dev) or PostgreSQL (prod) via SQLAlchemy

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Frontend (React)                   │
│   Vite · TypeScript · Tailwind · shadcn/ui · Zustand │
│                                                       │
│  Landing → Auth → Dashboard → NLP Workspace           │
│                            → Agentic Research         │
│                            → History · Reports        │
└────────────────────┬────────────────────────────────-┘
                     │ REST API (HTTP + WebSocket)
┌────────────────────▼────────────────────────────────-┐
│                   Backend (FastAPI)                   │
│        Uvicorn · SQLAlchemy · Pydantic v2             │
│                                                       │
│  /api/auth        JWT login & signup                  │
│  /api/documents   Upload, list, delete PDFs/TXTs      │
│  /api/nlp         TF-IDF, SBERT, K-Means, LDA         │
│  /api/research    LangGraph agentic swarm             │
└────────────────────┬────────────────────────────────-┘
                     │
        ┌────────────┴───────────┐
        │                        │
  SQLite / PostgreSQL         FAISS Vector Store
  (users, jobs, reports)      (per-research session)
```

### LangGraph Agent Pipeline

```
Planner (Llama 3.1 8B)
    └─► Researcher (Tavily Web Search)
            └─► Scraper (BS4 + FAISS)
                    └─► Synthesizer (Llama 3.3 70B)
                                └─► Reviewer (Llama 3.1 8B)
                                        └─► [loop if quality < threshold]
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm 9+

### 1. Clone

```bash
git clone https://github.com/Harsh-sh7/NLP-Research-Analyzer.git
cd NLP-Research-Analyzer
```

### 2. Configure API Keys

Copy the template env file inside the `backend/` folder and fill in your API keys:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env`:

```env
GROQ_API_KEY=your_groq_api_key_here       # https://console.groq.com/
TAVILY_API_KEY=tvly_your_key_here         # https://app.tavily.com/
SECRET_KEY=your_random_jwt_secret_here
```

> NLP analysis works without API keys. Only the Agentic Research Swarm requires `GROQ_API_KEY` and `TAVILY_API_KEY`.

### 3. Run the Backend

Create the virtual environment inside the `backend/` folder, install the requirements, and run the server:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt

# Run from the project root directory:
backend/.venv/bin/python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend is live at **http://localhost:8000**

### 4. Run the Frontend

Open a new terminal:

```bash
cd frontend
npm install
npm run dev
```

App is live at **http://localhost:5173**

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19, TypeScript, Vite 8, Tailwind CSS 3, shadcn/ui |
| **State** | Zustand |
| **Charts** | Recharts |
| **Graph** | ReactFlow (@xyflow/react) |
| **Animations** | Framer Motion |
| **Backend** | FastAPI, Uvicorn |
| **ORM** | SQLAlchemy 2, Pydantic v2 |
| **Auth** | JWT (python-jose + passlib bcrypt) |
| **NLP** | scikit-learn, NLTK, sentence-transformers, FAISS |
| **Agent** | LangGraph, LangChain, Groq (Llama), Tavily |
| **Database** | SQLite (dev) / PostgreSQL (prod) |

---

## 📁 Project Structure

```
NLP-Research-Analyzer/
├── README.md                       # Project overview & documentation
├── backend/
│   ├── .env.example                # API key template
│   ├── DEPLOY.md                   # Full deployment guide
│   ├── requirements.txt            # Python dependencies
│   └── app/
│       ├── main.py                 # FastAPI application entry point
│       ├── api/
│       │   ├── deps.py             # Auth dependency injection
│       │   └── endpoints/
│       │       ├── auth.py         # Signup / Login / Me
│       │       ├── documents.py    # Upload, list, delete, preload corpus
│       │       ├── nlp.py          # NLP analysis jobs
│       │       └── research.py     # Agentic research swarm jobs
│       ├── core/
│       │   ├── config.py           # Pydantic settings (reads .env)
│       │   └── security.py        # JWT creation & verification
│       ├── db/
│       │   ├── models.py           # SQLAlchemy ORM models
│       │   └── session.py          # DB engine & session factory
│       ├── schemas/                # Pydantic request/response schemas
│       │   ├── user.py
│       │   ├── document.py
│       │   ├── nlp.py
│       │   └── research.py
│       └── services/
│           ├── nlp_service.py      # TF-IDF, SBERT, K-Means, LDA logic
│           └── agent_service.py    # LangGraph swarm orchestration
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.ts
    ├── tailwind.config.js
    └── src/
        ├── App.tsx                 # Router & cursor glow effect
        ├── index.css               # Global styles & theme tokens
        ├── pages/
        │   ├── landing.tsx         # Hero landing page
        │   ├── auth.tsx            # Login / Signup
        │   ├── dashboard.tsx       # Document upload & job launcher
        │   ├── nlp.tsx             # NLP Workspace (heatmap, LDA, clusters…)
        │   ├── research.tsx        # Agentic Swarm + Knowledge Graph
        │   ├── history.tsx         # Job history
        │   ├── reports.tsx         # Saved research reports
        │   └── settings.tsx        # App preferences
        ├── components/ui/          # shadcn/ui component library
        ├── store/
        │   ├── authStore.ts        # User auth state (Zustand)
        │   ├── nlpStore.ts         # NLP job state
        │   ├── researchStore.ts    # Research job state
        │   └── uiStore.ts          # Theme & UI state
        └── lib/
            ├── api.ts              # Typed REST API client
            └── utils.ts            # Class merge utilities
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | For Agentic Research | Groq LLM API key — [console.groq.com](https://console.groq.com/) |
| `TAVILY_API_KEY` | For Agentic Research | Tavily web search key — [app.tavily.com](https://app.tavily.com/) |
| `SECRET_KEY` | Recommended | JWT signing secret (use a long random string in production) |
| `DATABASE_URL` | Optional | PostgreSQL URL (defaults to SQLite: `sqlite:///./sql_app.db`) |
| `REDIS_URL` | Optional | Redis URL for caching (disabled if not set) |

---

## 📖 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/signup` | Register a new user |
| `POST` | `/api/auth/login` | Login and receive JWT token |
| `GET` | `/api/auth/me` | Get current user profile |
| `GET` | `/api/documents/` | List uploaded documents |
| `POST` | `/api/documents/upload` | Upload PDF/TXT files |
| `DELETE` | `/api/documents/{id}` | Delete a document |
| `POST` | `/api/nlp/analyze` | Run NLP analysis on documents |
| `GET` | `/api/nlp/jobs` | List NLP job history |
| `GET` | `/api/nlp/jobs/{id}` | Get NLP job results |
| `POST` | `/api/research/` | Start an agentic research job |
| `GET` | `/api/research/jobs` | List research jobs |
| `GET` | `/api/research/jobs/{id}` | Get research job status & results |
| `GET` | `/api/research/reports` | List saved reports |
| `GET` | `/api/research/reports/{id}` | Get a specific report |

---

## 📄 License

MIT © [Harsh-sh7](https://github.com/Harsh-sh7)
