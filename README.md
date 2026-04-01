# Medical Protocol RAG System

A two-stage pipeline for scraping, parsing, and querying protocols from [diseases.medelement.com](https://diseases.medelement.com).

```
pipeline/   ← data engineering: scrape → parse → chunk → JSONL
rag/        ← inference: embed → index → retrieve → answer
config/     ← shared settings
```

---

## How It Works

1. **Pipeline** crawls protocol pages with Playwright, cleans HTML, converts to Markdown, extracts metadata (ICD-10 codes, specialty, year), and writes chunked records to `data/output/protocols.jsonl`.
2. **Indexer** reads the JSONL, embeds each chunk with a multilingual sentence-transformer model, and upserts vectors into Qdrant.
3. **Chat CLI** takes a question, retrieves the top-K relevant chunks from Qdrant, and sends them as context to an LLM (local Ollama or Anthropic Claude) for answer generation.

---

## Requirements

- Python 3.11+
- [Docker](https://www.docker.com/) — for Qdrant and Ollama
- [Playwright](https://playwright.dev/python/) browsers

```bash
pip install -r requirements.txt
playwright install chromium
```

Start infrastructure:

```bash
# Qdrant vector DB
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Ollama (local LLM)
docker run -d -p 11434:11434 --name ollama --gpus all ollama/ollama
docker exec -it ollama ollama pull llama3.2
```

---

## Configuration

Copy `.env.example` to `.env` and edit as needed:

```bash
cp .env.example .env
```

Key settings (all have defaults in `config/settings.py`):

| Variable | Default | Description |
|---|---|---|
| `CONCURRENCY` | `3` | Parallel protocol fetches |
| `REQUESTS_PER_SECOND` | `1.0` | Rate limit |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_MODEL` | `llama3.2` | Model for answer generation |
| `LLM_BACKEND` | `ollama` | `ollama` or `anthropic` |
| `ANTHROPIC_API_KEY` | _(empty)_ | Required only if using Anthropic |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `openai` |

---

## Usage

### 1. Run the scraping pipeline

```bash
# Fresh run
python -m pipeline.orchestrator

# Resume interrupted run
python -m pipeline.orchestrator --resume

# Limit to one specialty (for testing)
python -m pipeline.orchestrator --dry-run --specialties Кардиология
```

### 2. Index into Qdrant

```bash
python -m rag.indexer
# Recreate collection from scratch:
python -m rag.indexer --recreate
```

### 3. Chat

```bash
# Default: Ollama backend
python -m rag.chat

# Specific model or specialty filter
python -m rag.chat --model llama3.2 --category Кардиология

# Anthropic Claude backend
python -m rag.chat --backend anthropic --model claude-sonnet-4-6
```

Chat commands: `/quit` `/clear` `/sources`

---

## Project Structure

```
├── config/
│   ├── settings.py          # All settings (Pydantic, .env-aware)
│   └── specialties.py       # Medical specialty → content-type ID map
├── pipeline/
│   ├── orchestrator.py      # Entry point: python -m pipeline.orchestrator
│   ├── scraper/             # Playwright-based listing + protocol scrapers
│   ├── parser/              # HTML cleaning, Markdown conversion, metadata, OCR
│   ├── chunker/             # Text splitting + context header injection
│   ├── state_manager.py     # SQLite-backed resumable state (pending/done/failed)
│   └── writer.py            # JSONL output writer
├── rag/
│   ├── indexer.py           # Entry point: python -m rag.indexer
│   ├── chat.py              # Entry point: python -m rag.chat
│   ├── retriever.py         # Qdrant similarity search
│   └── embeddings.py        # Local (sentence-transformers) or OpenAI embedder
├── tests/
│   ├── pipeline/            # Unit tests for parser, chunker
│   └── rag/                 # Unit tests for retriever
├── data/
│   └── output/              # protocols.jsonl written here (gitignored)
├── requirements.txt
└── reset_state.py           # Resets state.db to re-scrape all protocols
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests are mocked — no live Qdrant or browser required. Tests requiring `qdrant-client` are auto-skipped if the package is not installed.

---

## Output Format

Each line in `protocols.jsonl` is one chunk:

```json
{
  "document_id": "hellp-17522",
  "source_url": "https://diseases.medelement.com/disease/hellp/17522",
  "protocol_name": "HELLP-синдром",
  "year": 2023,
  "category": "Акушерство и гинекология",
  "icd_codes": ["O14.2"],
  "section_name": "Диагностика",
  "chunk_index": 2,
  "text": "[Protocol: HELLP-синдром, 2023] [Section: Диагностика]\n\n..."
}
```

---

## Resetting the Pipeline

To re-scrape everything from scratch:

```bash
python reset_state.py
rm data/output/protocols.jsonl
python -m pipeline.orchestrator --resume
```
