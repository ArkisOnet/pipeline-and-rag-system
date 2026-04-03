# Medical Protocol RAG System

A two-stage pipeline for scraping, parsing, and querying Kazakhstan clinical protocols from [diseases.medelement.com](https://diseases.medelement.com), enriched with a real-world medical services registry.

```
pipeline/   ← data engineering: scrape → parse → chunk → JSONL
            └── parser/services_parser.py  ← services.xlsx ingestion
rag/        ← inference: embed → index → retrieve → answer
config/     ← shared settings (.env-aware)
```

---

## How It Works

1. **Pipeline** crawls protocol pages with Playwright, cleans HTML, converts tables to Markdown, extracts metadata (ICD-10 codes, specialty, year), and writes chunked records to `data/output/protocols.jsonl`.
2. **Services Parser** reads `data/raw/services.xlsx` (medical services registry) — each row becomes one discrete vector record (no chunking).
3. **Indexers** embed and upsert both data sources into separate Qdrant collections:
   - `medprotocols` — clinical protocol chunks
   - `medservices` — service registry rows
4. **Chat CLI** detects ICD-10 codes in the user query, performs exact-match retrieval first (priority), then semantic search above a similarity threshold (0.70). Results from both collections are merged, deduplicated by `Код услуги`, and sent as context to the medical LLM API.

---

## Requirements

- Python 3.11+
- [Docker](https://www.docker.com/) — for Qdrant
- [Playwright](https://playwright.dev/python/) browsers

```bash
pip install -r requirements.txt
playwright install chromium
```

Start Qdrant:

```bash
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
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
| `API_KEY` | _(required)_ | Medical LLM API key |
| `CONCURRENCY` | `3` | Parallel protocol fetches |
| `REQUESTS_PER_SECOND` | `1.0` | Crawl rate limit |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `medprotocols` | Protocols collection name |
| `SERVICES_COLLECTION` | `medservices` | Services collection name |
| `SERVICES_XLSX_PATH` | `data/raw/services.xlsx` | Path to services file |
| `MIN_SIMILARITY_SCORE` | `0.70` | Minimum cosine similarity threshold |
| `TOP_K` | `5` | Results per source per query |
| `EMBEDDING_MODEL_LOCAL` | `paraphrase-multilingual-mpnet-base-v2` | Multilingual embedder |
| `ANTHROPIC_API_KEY` | _(empty)_ | Optional — Anthropic fallback |

---

## Usage

### 1. Run the scraping pipeline

```bash
# Fresh run
python -m pipeline.orchestrator

# Resume an interrupted run
python -m pipeline.orchestrator --resume

# Test with one specialty only
python -m pipeline.orchestrator --dry-run --specialties Кардиология
```

### 2. Index clinical protocols into Qdrant

```bash
python -m rag.indexer
# Recreate collection from scratch:
python -m rag.indexer --recreate
```

### 3. Index services registry into Qdrant

```bash
python -m rag.indexer_services
# Recreate:
python -m rag.indexer_services --recreate
```

### 4. Chat

```bash
python -m rag.chat

# Filter protocols by specialty
python -m rag.chat --category Кардиология

# Pass API key on the command line
python -m rag.chat --api-key YOUR_KEY
```

Chat commands: `/quit` `/clear` `/sources`

---

## Retrieval Logic

When the user asks a question:

1. **ICD-10 detection** — the query is scanned for codes matching `[A-Z]\d{2}(\.\d{1,2})?` (e.g. `Z35.8`, `I21.0`).
2. **Exact-match first** — if a code is found, Qdrant is filtered on `icd_codes` (protocols) and `icd_code` (services) to guarantee relevant records surface at the top (score = 1.0).
3. **Semantic search** — vector similarity search runs in parallel; any result with score < `MIN_SIMILARITY_SCORE` (0.70) is dropped.
4. **Merge & deduplicate** — exact matches prepend the semantic results; services are deduplicated by `Код услуги` so the same service never appears twice.
5. **No-results handling** — if nothing passes the threshold, the bot responds: *"К сожалению, в базе нет данных по этому коду МКБ-10"* instead of showing unrelated protocols.

---

## Response Format

Each answer is structured as:

```
Assistant: <clinical answer from LLM>

**Реестр медицинских услуг:**
| № | Услуга | Код услуги | МКБ-10 | Диагноз | Цена | Место оказания |
|---|--------|------------|--------|---------|------|----------------|
| 1 | ...    | ...        | ...    | ...     | ...  | ...            |

---
**Источники:**
*Клинические протоколы МЗ РК*
  [1] Protocol Name — Section  <url>
*Реестр медицинских услуг*
  N запись(-ей) из базы услуг (таблица выше)
```

---

## Project Structure

```
├── config/
│   ├── settings.py               # All settings (Pydantic, .env-aware)
│   └── specialties.py            # Medical specialty → content-type ID map
├── pipeline/
│   ├── orchestrator.py           # Entry point: python -m pipeline.orchestrator
│   ├── scraper/                  # Playwright listing + protocol scrapers
│   ├── parser/
│   │   ├── html_cleaner.py       # HTML → clean Markdown (tables, images, OCR)
│   │   ├── markdown_converter.py
│   │   ├── metadata_extractor.py # ICD-10, year, specialty extraction
│   │   ├── image_ocr.py          # Optional easyocr for scanned tables
│   │   └── services_parser.py    # services.xlsx → list of dicts (one per row)
│   ├── chunker/                  # Text splitting + context header injection
│   ├── state_manager.py          # SQLite-backed resumable state
│   └── writer.py                 # JSONL output writer
├── rag/
│   ├── indexer.py                # Index protocols: python -m rag.indexer
│   ├── indexer_services.py       # Index services: python -m rag.indexer_services
│   ├── chat.py                   # Chat CLI: python -m rag.chat
│   ├── retriever.py              # ICD-aware retrieval + score threshold + dedup
│   ├── embeddings.py             # Local (sentence-transformers) or OpenAI embedder
│   └── prompts.py                # System prompt for the medical assistant persona
├── tests/
│   ├── pipeline/                 # Unit tests: parser, chunker
│   └── rag/                      # Unit tests: retriever
├── data/
│   ├── raw/                      # services.xlsx goes here (not committed)
│   └── output/                   # protocols.jsonl written here (not committed)
├── requirements.txt
└── reset_state.py                # Resets state.db to re-scrape all protocols
```

---

## Data Sources

| Source | Collection | Dedup key | ICD field |
|--------|-----------|-----------|-----------|
| `protocols.jsonl` (scraped) | `medprotocols` | text prefix | `icd_codes` (array) |
| `services.xlsx` (registry) | `medservices` | `Код услуги` | `icd_code` (string) |

---

## Running Tests

```bash
pytest tests/ -v
```

Tests are mocked — no live Qdrant or browser required. Tests requiring `qdrant-client` are auto-skipped if the package is not installed.

---

## Output Format — protocols.jsonl

Each line is one chunk:

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

```bash
python reset_state.py
rm data/output/protocols.jsonl
python -m pipeline.orchestrator --resume
```
