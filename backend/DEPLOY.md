# 🚀 NLP Research Analyzer — Deployment Guide

A full-stack AI research platform with a **FastAPI backend** and **React + Vite frontend**.

---

## 📋 Prerequisites

| Tool | Min Version | Check |
|------|------------|-------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | any | `git --version` |

> **Optional (for production):** PostgreSQL 14+, Redis 7+

---

## ⚙️ 1. Clone the Repository

```bash
git clone https://github.com/Harsh-sh7/NLP-Research-Analyzer.git
cd NLP-Research-Analyzer
```

---

## 🔑 2. Configure Environment Variables

Copy the example env file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` with the following required keys:

```env
# --- Required for Agentic Research Mode ---

# Groq LLM (free tier available)
# Get yours at: https://console.groq.com/
GROQ_API_KEY=your_groq_api_key_here

# Tavily Web Search (free tier available)
# Get yours at: https://app.tavily.com/
TAVILY_API_KEY=tvly_your_tavily_api_key_here

# --- Optional (security) ---
SECRET_KEY=a_very_long_random_secret_string_for_jwt_signing

# --- Optional (if using PostgreSQL instead of SQLite) ---
# DATABASE_URL=postgresql://user:password@localhost:5432/nlp_db

# --- Optional (if using Redis for caching) ---
# REDIS_URL=redis://localhost:6379
```

> **Note:** Without the above API keys, the Agentic Research (Swarm) feature will not work. All other NLP features (document analysis, LDA, clustering, etc.) work without any API keys.

---

## 🐍 3. Backend Setup (FastAPI)

### 3a. Create & activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3b. Install Python dependencies

```bash
pip install -r backend/requirements.txt
```

> ⏳ First install may take a few minutes (downloads ML models like `sentence-transformers`).

### 3c. Run the backend server

```bash
# From the project root directory
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be live at **http://localhost:8000**  
Interactive API docs: **http://localhost:8000/api/openapi.json**

---

## ⚛️ 4. Frontend Setup (React + Vite)

Open a **new terminal** (keep the backend running):

```bash
cd frontend
npm install
npm run dev
```

The app will open at **http://localhost:5173**

---

## 🌐 5. Using the App

1. Navigate to **http://localhost:5173**
2. Click **Get Started** on the landing page
3. **Sign up** for a new account (username + email + password)
4. Upload research documents (PDF or TXT) from the Dashboard
5. Run **NLP Analysis** (LDA, clustering, similarity heatmap)
6. Use the **Agentic Research** swarm to auto-research any topic

---

## 🐳 6. Docker Deployment (Optional)

If you want to containerize the app, create a `docker-compose.yml`:

```yaml
version: "3.9"
services:
  backend:
    build:
      context: .
      dockerfile: Dockerfile.backend
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./sql_app.db:/app/sql_app.db
      - ./backend/uploads:/app/backend/uploads

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "5173:80"
    depends_on:
      - backend
```

And a `Dockerfile.backend`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

And a `frontend/Dockerfile`:

```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
```

Run with:

```bash
docker compose up --build
```

---

## ☁️ 7. Cloud Deployment Options

### Option A: Deploy Frontend on Vercel (recommended)

1. Push your code to GitHub (already done ✅)
2. Go to [vercel.com](https://vercel.com) → **New Project** → Import from GitHub
3. Set **Root Directory** to `frontend`
4. Set **Build Command** to `npm run build`
5. Set **Output Directory** to `dist`
6. Add environment variable: `VITE_API_URL=https://your-backend-domain.com/api`
7. Click **Deploy**

> Update `frontend/src/lib/api.ts` line 1 to use the env var:
> ```ts
> const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000/api"
> ```

### Option B: Deploy Backend on Railway / Render (Free Tier Optimized)

Because Render and Railway free tiers are constrained to **512 MB of RAM**, installing standard machine learning dependencies like `sentence-transformers` (which pulls in PyTorch with CUDA) will cause **Out Of Memory (OOM) build/runtime crashes**.

To prevent this, use the optimized production requirements file:

1. Go to [render.com](https://render.com) or [railway.app](https://railway.app).
2. Create a new **Web Service** from your GitHub repo.
3. Configure the following build settings:
   - **Root Directory**: `.` (project root)
   - **Build Command**: `pip install -r backend/requirements-prod.txt` (This skips PyTorch/SBERT, using the lightweight TF-IDF LSA fallback for document analysis instead, keeping RAM usage < 100MB).
   - **Start Command**: `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT`
4. Add all your environment variables (GROQ_API_KEY, TAVILY_API_KEY, SECRET_KEY, etc.).
5. For PostgreSQL, provision a managed database from the same platform and use its connection URL as `DATABASE_URL`.

### Option C: Deploy on a VPS (Ubuntu)

```bash
# Install dependencies
sudo apt update && sudo apt install python3.11 python3-pip nodejs npm nginx -y

# Clone and setup
git clone https://github.com/Harsh-sh7/NLP-Research-Analyzer.git
cd NLP-Research-Analyzer
cp .env.example .env
nano .env  # fill in your API keys

# Backend
python3 -m venv venv && source venv/bin/activate
pip install -r backend/requirements.txt

# Run backend as a service (using systemd)
sudo nano /etc/systemd/system/nlp-backend.service
```

Paste this into the systemd service file:

```ini
[Unit]
Description=NLP Research Analyzer Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/NLP-Research-Analyzer
ExecStart=/home/ubuntu/NLP-Research-Analyzer/venv/bin/uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
Restart=always
EnvironmentFile=/home/ubuntu/NLP-Research-Analyzer/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable nlp-backend && sudo systemctl start nlp-backend

# Frontend
cd frontend && npm install && npm run build
sudo cp -r dist/* /var/www/html/

# Configure nginx
sudo nano /etc/nginx/sites-available/nlp
```

Paste nginx config:

```nginx
server {
    listen 80;
    server_name your-domain.com;

    root /var/www/html;
    index index.html;

    # Serve frontend
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Proxy backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/nlp /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🔒 8. Production Security Checklist

- [ ] Change `SECRET_KEY` in `.env` to a long random string (`openssl rand -hex 32`)
- [ ] Use PostgreSQL instead of the default SQLite for production
- [ ] Enable HTTPS with Let's Encrypt (`certbot --nginx`)
- [ ] Update `BACKEND_CORS_ORIGINS` in `backend/app/core/config.py` to only allow your frontend domain
- [ ] Do NOT commit `.env` to Git (it is already in `.gitignore`)
- [ ] Store uploaded files in persistent cloud storage (e.g., AWS S3) instead of local disk for multi-replica deployments

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'backend'` | Run uvicorn from the **project root**, not inside `backend/` |
| Frontend can't reach backend | Check `API_BASE` in `frontend/src/lib/api.ts` matches your backend URL |
| `GROQ_API_KEY not set` error | Add the key to your `.env` file and restart the backend |
| SQLite locked error | Don't run multiple backend workers with SQLite; switch to PostgreSQL |
| NLP analysis is slow | The first run downloads `sentence-transformers` models; subsequent runs are faster |
| Port 8000 already in use | `lsof -ti:8000 \| xargs kill -9` then restart the backend |

---

## 📁 Project Structure

```
NLP-Research-Analyzer/
├── .env.example           # Environment variable template
├── DEPLOY.md              # This file
├── backend/
│   ├── requirements.txt   # Python dependencies
│   └── app/
│       ├── main.py        # FastAPI entry point
│       ├── api/           # REST endpoints (auth, documents, nlp, research)
│       ├── core/          # Config, JWT security
│       ├── db/            # SQLAlchemy models & session
│       ├── schemas/       # Pydantic request/response schemas
│       └── services/      # NLP & Agentic research logic
└── frontend/
    ├── package.json       # Node dependencies
    ├── vite.config.ts     # Vite bundler config
    └── src/
        ├── pages/         # React pages (landing, dashboard, nlp, research…)
        ├── components/    # Reusable UI components
        ├── store/         # Zustand global state
        └── lib/           # API client & utilities
```

---

*Built with ❤️ using FastAPI, React, LangGraph, and Groq LLMs.*
