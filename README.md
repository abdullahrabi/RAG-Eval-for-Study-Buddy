# StudyBuddy RAG Evaluation System

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![TruLens](https://img.shields.io/badge/TruLens-powered-orange.svg)

---

## Problem & Solution

**StudyBuddy RAG Evaluation System** solves the critical challenge of **continuously measuring RAG quality in production** for a personalized AI study assistant. Traditional RAG evaluation is either one-off, manual, or lacks context-aware metrics. This system provides **automated, continuous, per-user evaluation** with five context-aware metrics (Answer Relevance, Answer Quality, Context Relevance, Groundedness, Correctness) powered by TruLens and Cloudflare Workers AI. It discovers users from Pinecone, generates evaluation questions from their notes, re-uses existing questions across runs, and feeds results into a live TruLens dashboard — enabling data-driven iteration on retrieval, prompts, and model selection without engineering overhead.

---

## Architecture Diagram

```mermaid
flowchart TB
    subgraph "Data Layer"
        PC[(Pinecone Vector DB\nUsers + Notes)]
        TS[(TruLens SQLite\nEvaluation Records)]
    end

    subgraph "Embedding & Retrieval"
        GE[Gemini Embedding 2\n768-dim + Task Prefix]
        RT[OptimizedRAG\nTop-K=3 Retrieval]
    end

    subgraph "LLM Providers"
        GQ[Groq\nGPT-OSS 20B/120B]
        CF[Cloudflare Workers AI\n@cf/gpt-oss-120b]
    end

    subgraph "Evaluation Pipeline"
        QG[Question Generator\nPer-User Caching]
        EV[Evaluator\n5 Context-Aware Metrics]
        TL[TruLens Instrumentation\nOTEL Spans]
    end

    subgraph "Observability"
        DB[TruLens Dashboard\nStreamlit on :8080]
    end

    PC -->|User Discovery| QG
    PC -->|Notes Fetch| RT
    GE -->|Query Embedding| RT
    GQ -->|RAG Generation| RT
    GQ -->|Question Gen| QG
    CF -.->|Primary: 5 Metrics| EV
    GQ -.->|Fallback: 5 Metrics| EV
    RT -->|Instrumented Query| TL
    TL -->|Spans + Feedback| TS
    TS -->|Live Data| DB
    QG -->|Questions| EV
    EV -->|Scores| TS

    style PC fill:#f9f,stroke:#333
    style TS fill:#bbf,stroke:#333
    style CF fill:#fd0,stroke:#333
    style DB fill:#0f0,stroke:#333
```

---

## Agent / RAG Workflow

```mermaid
sequenceDiagram
    participant User
    participant Entrypoint
    participant Dashboard
    participant Evaluator
    participant Pinecone
    participant TruLens
    participant Groq
    participant Cloudflare
    participant Gemini

    Entrypoint->>Dashboard: Start Streamlit Dashboard (:8080)
    Entrypoint->>Evaluator: Launch RAG_Eval.py
    Evaluator->>Pinecone: Discover Users (namespace: users)
    Evaluator->>Pinecone: Fetch Notes per User (namespace: notes)
    Evaluator->>Gemini: Embed Questions (task prefix)
    Evaluator->>Groq: Generate Questions from Notes
    Note over Evaluator: Per-user question caching\n(questions/questions_<id>.txt)
    Evaluator->>TruLens: Load Existing Questions (skip if evaluated)
    loop For Each New Question
        Evaluator->>Groq: RAG Query (retrieve + generate)
        Groq->>Pinecone: Vector Search (top_k=3)
        Pinecone-->>Groq: Context Chunks
        Groq-->>Evaluator: Answer
        Evaluator->>TruLens: Record Span (OTEL)
        par Context-Aware Evaluation
            Evaluator->>Cloudflare: Answer Relevance
            Evaluator->>Cloudflare: Answer Quality (+context)
            Evaluator->>Cloudflare: Context Relevance
            Evaluator->>Cloudflare: Groundedness (+context)
            Evaluator->>Cloudflare: Correctness (+context)
        or Fallback (no CF creds)
            Evaluator->>Groq: All 5 Metrics via LiteLLM
        end
        Cloudflare-->>TruLens: Feedback Scores
        Groq-->>TruLens: Feedback Scores
    end
    Evaluator->>TruLens: Force Flush
    Dashboard->>TruLens: Live Query Results
    User->>Dashboard: View Metrics & Traces
```

---

## Features

| Category | Details |
|----------|---------|
| **Evaluation Metrics** | 5 context-aware metrics: Answer Relevance, Answer Quality, Context Relevance, Groundedness, Correctness |
| **Context-Aware Judging** | Correctness & Answer Quality receive retrieved context → eliminates short-answer penalization |
| **Per-User Question Caching** | Questions saved to `questions/questions_<user_id>.txt`; re-used across runs |
| **Skip Existing** | Already-evaluated questions read from TruLens DB and skipped automatically |
| **Auto Top-K Scaling** | `calculate_top_k()` adapts retrieval width to corpus size (3–40) |
| **Multi-Provider Evaluation** | Primary: Cloudflare Workers AI (`@cf/openai/gpt-oss-120b`); Fallback: Groq via LiteLLM |
| **Live Dashboard** | TruLens Streamlit dashboard runs alongside evaluation — real-time metrics |
| **OTEL Instrumentation** | Full span tracing for retrieval + generation via `trulens.otel` |
| **Graceful Shutdown** | SIGINT/SIGTERM handling keeps dashboard running after evaluation completes |

---

## Technology Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.11+ |
| **Vector Store** | Pinecone (serverless) |
| **Embeddings** | Google Gemini Embedding 2 (768-dim, task-prefixed) |
| **RAG Generation** | Groq — `openai/gpt-oss-20b` |
| **Question Generation** | Groq — `openai/gpt-oss-120b` (fallback: `moonshotai/kimi-k2-instruct`) |
| **Evaluation (Primary)** | Cloudflare Workers AI — `@cf/openai/gpt-oss-120b` |
| **Evaluation (Fallback)** | Groq via LiteLLM — `openai/gpt-oss-20b` & `openai/gpt-oss-120b` |
| **Evaluation Framework** | TruLens Core + Providers + Dashboard |
| **Observability** | OpenTelemetry (OTEL) spans |
| **Dashboard** | Streamlit (TruLens built-in) |
| **Orchestration** | Custom `entrypoint.py` (dashboard + evaluator processes) |
| **Containerization** | Docker (python:3.11-slim) |
| **Config** | `.env` via `python-dotenv` |

---

## Screenshots

> **Add your screenshots here** — replace the placeholder paths with actual images in `docs/screenshots/`

| Dashboard Overview | Metric Drill-Down | Trace View |
|:---:|:---:|:---:|
| ![Dashboard](docs/screenshots/dashboard-overview.png) | ![Metrics](docs/screenshots/metric-detail.png) | ![Traces](docs/screenshots/trace-view.png) |
| *Live TruLens dashboard showing per-user app versions* | *Five context-aware metrics with reasons* | *OTEL spans: retrieval → generation → feedback* |

### Video Demo (2–3 min)

> **Record and add a demo video** — save as `docs/demo.mp4` or host externally

[![Demo Video](docs/screenshots/demo-thumbnail.png)](docs/demo.mp4)

*Click to watch the 2-minute walkthrough: user discovery → question generation → live evaluation → dashboard exploration*

---

## Local Setup

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized run)
- API keys for:
  - **Gemini** (embeddings) — [Google AI Studio](https://aistudio.google.com/)
  - **Pinecone** (vector DB) — [Pinecone Console](https://app.pinecone.io/)
  - **Groq** (RAG + question gen + fallback eval) — [Groq Console](https://console.groq.com/)
  - **Cloudflare Workers AI** (primary evaluation) — [Cloudflare Dashboard](https://dash.cloudflare.com/) → Workers AI → API Token + Account ID

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd "RAG Eval"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

**`.env.example`** (copy to `.env` and fill in):

```env
# ─── Required ──────────────────────────────────────────────
GEMINI_API_KEY=your_gemini_api_key
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key

# ─── Cloudflare Workers AI (Primary Evaluation) ────────────
CLOUDFLARE_API_KEY-2=your_cloudflare_api_token
CLOUDFLARE_ACCOUNT_ID-2=your_cloudflare_account_id

# ─── Pinecone Index ────────────────────────────────────────
INDEX_NAME=studybuddy

# ─── TruLens Database ──────────────────────────────────────
TRULENS_DB_URL=sqlite:///data/default.sqlite
TRULENS_APP_NAME=StudyBuddy RAG Evaluation
TRULENS_APP_VERSION=v3

# ─── Model Selection (override if needed) ──────────────────
RAG_MODEL=openai/gpt-oss-20b
QUESTION_MODEL=openai/gpt-oss-120b
FAST_EVAL_MODEL=openai/gpt-oss-20b
DEEP_EVAL_MODEL=openai/gpt-oss-120b
CLOUDFLARE_MODEL=@cf/openai/gpt-oss-120b
FALLBACK_QUESTION_MODEL=moonshotai/kimi-k2-instruct
EMBEDDING_MODEL=gemini-embedding-2
EMBEDDING_DIMENSION=768

# ─── Evaluation Behavior ───────────────────────────────────
QUESTIONS_PER_USER=25
INTER_QUESTION_DELAY=15
SKIP_EXISTING_QUESTIONS=true

# ─── Token Limits ──────────────────────────────────────────
RAG_MAX_TOKENS=2000
QUESTION_MAX_TOKENS=2500
CLOUDFLARE_MAX_TOKENS=500

# ─── Server ────────────────────────────────────────────────
PORT=8080
QUESTIONS_DIR=questions
```

### 3. Run Locally

```bash
# Option A: Run dashboard + evaluator together (recommended)
python entrypoint.py

# Option B: Run separately
# Terminal 1:
python Dashboard.py
# Terminal 2 (after dashboard is up):
python RAG_Eval.py
```

- **Dashboard**: http://localhost:8080
- **Evaluation logs**: stdout in terminal

---

## Docker Deployment

### Build

```bash
docker build -t studybuddy-rag-eval .
```

### Run

```bash
docker run -d \
  --name rag-eval \
  -p 8080:8080 \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/questions:/app/questions \
  studybuddy-rag-eval
```

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: "3.8"

services:
  rag-eval:
    build: .
    container_name: studybuddy-rag-eval
    ports:
      - "8080:8080"
    env_file: .env
    volumes:
      - ./data:/app/data
      - ./questions:/app/questions
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
docker compose up -d
docker compose logs -f
```

### Production Notes

- Use a managed PostgreSQL for TruLens DB in production:  
  `TRULENS_DB_URL=postgresql://user:pass@host:5432/trulens`
- Mount persistent volumes for `data/` and `questions/`
- Run behind a reverse proxy (nginx/Traefik) with TLS
- Set resource limits: `deploy.resources.limits.memory: 2G`

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install ruff pytest pytest-cov
      
      - name: Lint (ruff)
        run: ruff check .
      
      - name: Type check (mypy)
        run: mypy --ignore-missing-imports RAG_Eval.py Dashboard.py entrypoint.py
      
      - name: Unit tests
        run: pytest tests/ -v --cov=. --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml

  build-and-push:
    needs: lint-and-test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Log in to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=sha
            type=raw,value=latest,enable={{is_default_branch}}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy-staging:
    needs: build-and-push
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment: staging
    steps:
      - name: Deploy to Staging
        run: |
          echo "Deploy to staging environment..."
          # kubectl set image, docker compose pull, or your deployment method

  deploy-production:
    needs: build-and-push
    if: github.event_name == 'release'
    runs-on: ubuntu-latest
    environment: production
    steps:
      - name: Deploy to Production
        run: |
          echo "Deploy to production..."
          # Blue/green, canary, or your strategy
```

### Required Secrets (GitHub Settings → Secrets)

| Secret | Description |
|--------|-------------|
| `GEMINI_API_KEY` | Google AI Studio key |
| `PINECONE_API_KEY` | Pinecone API key |
| `GROQ_API_KEY` | Groq API key |
| `CLOUDFLARE_API_KEY-2` | Cloudflare Workers AI token |
| `CLOUDFLARE_ACCOUNT_ID-2` | Cloudflare account ID |
| `DOCKERHUB_USERNAME` / `GITHUB_TOKEN` | Container registry auth |

---

## Testing

### Run Unit Tests

```bash
# Install test deps
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest tests/ -v --cov=. --cov-report=term-missing
```

### Test Structure

```
tests/
├── unit/
│   ├── test_embedding.py          # GeminiDirectEmbedding
│   ├── test_question_gen.py       # generate_questions_from_notes
│   ├── test_question_cache.py     # parse/save question files
│   ├── test_top_k_scaling.py      # calculate_top_k()
│   ├── test_metrics.py            # CloudflareProvider metric methods
│   └── test_rag_app.py            # OptimizedRAG retrieve/query
├── integration/
│   ├── test_pinecone_flow.py      # User discovery + notes fetch
│   └── test_trulens_flow.py       # TruSession + feedback recording
└── conftest.py                    # Fixtures, mock clients
```

### Example Unit Test

```python
# tests/unit/test_top_k_scaling.py
import pytest
from RAG_Eval import calculate_top_k

@pytest.mark.parametrize("chunks,expected", [
    (10, 3),
    (30, 5),
    (75, 8),
    (200, 15),
    (500, 25),
    (1000, 40),
])
def test_calculate_top_k(chunks, expected):
    assert calculate_top_k(chunks) == expected
```

### Mocking External APIs

Use `pytest-mock` to mock `Pinecone`, `Groq`, `OpenAI`, `google.genai`, and `TruSession`:

```python
# tests/conftest.py
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_pinecone():
    return MagicMock()

@pytest.fixture
def mock_groq():
    return MagicMock()

@pytest.fixture
def mock_trulens_session():
    session = MagicMock()
    session.get_records_and_feedback.return_value = (None, None)
    return session
```

---

## Security Considerations

| Area | Mitigation |
|------|------------|
| **API Keys** | Never commit `.env`; use `.env.example` template; inject via CI secrets or vault |
| **Network** | Run in private VPC; restrict Pinecone/Cloudflare/Groq to allowlisted IPs |
| **Data Privacy** | User notes never leave Pinecone → Groq/Cloudflare; no PII in evaluation prompts |
| **Least Privilege** | Pinecone API key: read-only on `notes` + `users` namespaces; Cloudflare token: Workers AI read only |
| **Container** | Non-root user in Dockerfile; read-only rootfs; drop ALL capabilities |
| **Dashboard Auth** | TruLens dashboard has no built-in auth — put behind VPN/SSO/oauth2-proxy |
| **Secrets in Logs** | `RAG_Eval.py` masks keys with ✅/❌; never logs raw values |
| **Dependencies** | Pin versions in `requirements.txt`; run `pip-audit` in CI |

### Hardened Dockerfile (Production)

```dockerfile
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH

# Non-root user
RUN useradd --no-create-home --uid 10000 appuser \
 && chown -R appuser:appuser /app
USER appuser

# Security hardening
# (Add read-only rootfs, drop caps via runtime, not Dockerfile)

EXPOSE 8080
CMD ["python", "entrypoint.py"]
```

---

## Limitations & Known Issues

| Limitation | Workaround / Future Work |
|------------|--------------------------|
| **Single SQLite DB** | TruLens SQLite doesn't support concurrent writers — run one evaluator at a time; migrate to PostgreSQL for scale |
| **No Dashboard Auth** | Deploy behind oauth2-proxy, Cloudflare Access, or VPN |
| **Cloudflare Rate Limits** | Workers AI has per-account limits; fallback to Groq activates automatically |
| **Groq TPM Limits** | Fallback path may hit tokens-per-minute; add exponential backoff if needed |
| **Question Quality** | AI-generated questions depend on note quality; consider human-in-the-loop review |
| **Fixed Top-K=3** | Locked after empirical testing; `calculate_top_k()` exists but unused — enable if corpus varies widely |
| **No Incremental Embedding** | New notes require re-embedding; integrate Pinecone upsert webhook for real-time |
| **Single RAG Prompt** | Prompt tuned for educational QA; may need adaptation for other domains |
| **Windows Paths** | `QUESTIONS_DIR` uses forward slashes; works on Windows but test Docker volume mounts |

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/amazing-feature`
3. Run tests: `pytest tests/ -v`
4. Lint: `ruff check . && mypy RAG_Eval.py`
5. Commit: `git commit -m "feat: add amazing feature"`
6. Push: `git push origin feat/amazing-feature`
7. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [TruLens](https://www.trulens.org/) — Evaluation framework & dashboard
- [Groq](https://groq.com/) — Fast inference for generation & fallback eval
- [Cloudflare Workers AI](https://developers.cloudflare.com/workers-ai/) — Cost-effective evaluation models
- [Pinecone](https://www.pinecone.io/) — Serverless vector database
- [Google Gemini](https://ai.google.dev/) — Embedding model
- [Streamlit](https://streamlit.io/) — Dashboard UI