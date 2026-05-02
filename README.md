# ocr-kb — Local OCR Knowledge Base

A fully local, privacy-preserving research knowledge base. Drop in PDFs, images,
or text files; extract structured text and store everything in SQLite with full-text
search. Ask questions in plain English and get cited answers — all on your own
hardware, no cloud required.

**Key properties:**

- **granite-docling by default** — [ibm-granite/granite-docling-258M](https://huggingface.co/ibm-granite/granite-docling-258M) is the primary OCR engine; it handles both digital and scanned PDFs locally with no external server required. Falls back to CPU automatically if CUDA VRAM is occupied by another process (e.g. Ollama)
- All inference runs locally — no API keys, no data leaves your machine
- Optional Ollama / LM Studio backend for text enrichment (summaries, key points, RAG answers) and embedded figure captioning
- **Structured markdown output** — headings, tables, formulas, and lists are preserved as proper markdown, not flattened text
- **Automatic metadata extraction** — title, authors, year, DOI, abstract, and journal are parsed from the first page on ingest; uses the free Crossref API for DOI-indexed papers, falls back to LLM extraction otherwise
- **SPECTER2 embeddings (default)** — [allenai/specter2](https://huggingface.co/allenai/specter2_base) asymmetric embeddings purpose-built for academic paper retrieval; uses a separate adapter for queries vs. documents for higher precision
- **Preprint-aware extraction** — automatically detects and strips sequential line numbers added by medRxiv/bioRxiv for reviewer comments, producing clean text regardless of preprint format
- Per-run model override — switch models for a single ingest or ask without changing config
- Deduplication on re-ingest — same file updated in-place, existing tags/category preserved
- **Hybrid semantic + keyword search** — FTS5 full-text search merged with embeddings via Reciprocal Rank Fusion; filter by author, year, DOI, tag, or category
- RAG question-answering with inline citations and source map
- **Observability** — structured logging of ingest speed, search latency, and errors to `kb_data/logs/`
- **Error recovery** — failed documents are tracked in a dead-letter queue with retry logic
- Wiki synthesis — auto-generates a linked concept graph from the entire KB
- MCP server — expose all 12 KB operations to Claude Code, Cursor, or any MCP-aware agent

---

## Hardware requirements

**OCR (granite-docling) — no server required:**

| Scenario | Device | Notes |
|----------|--------|-------|
| GPU with free VRAM | `DOCLING_DEVICE=cuda` (default) | ~2–3 GB VRAM; fast |
| GPU occupied by Ollama | automatic CPU fallback | slower but still works |
| CPU only | `DOCLING_DEVICE=cpu` | works, ~2–5× slower |

**Optional enrichment + RAG (Ollama or LM Studio):**

| Slot | Role | Recommended model | VRAM |
|------|------|-------------------|------|
| Text (enrichment + RAG) | Summaries, key points, RAG answers | `gemma3:4b`, `llama3.2:3b` | ~2–3 GB |
| Vision (figure captions) | Per-figure descriptions (`EXTRACT_EMBEDDED_IMAGES=true`) | `llava:7b`, `qwen2.5vl:7b` | ~5–6 GB |

Enrichment and RAG are optional — basic ingest and search work without any Ollama model loaded.

---

## Installation

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the package with docling (primary OCR engine) + SPECTER2 embeddings
pip install -e ".[dev,docling,specter2]"

# specter2 adds the 'adapters' library for asymmetric academic embeddings.
# If you skip it, embeddings fall back to all-MiniLM-L6-v2 automatically.

# 3. Verify
ocr-kb --help
```

Python 3.11 or later is required. `docling` pulls in `torch` and `transformers` — the first `ocr-kb ingest` will download the granite-docling-258M weights (~500 MB) from HuggingFace and cache them locally.

---

## Ollama setup (recommended)

1. Install [Ollama](https://ollama.com).
2. Pull the models you want to use:

```bash
# RTX 4070 12 GB — comfortable dual-model setup
ollama pull llava:7b          # vision: OCR + figure description
ollama pull gemma3:4b         # text: enrichment + RAG

# Better vision quality (needs ~7 GB)
ollama pull qwen2.5vl:7b

# Faster text model (needs ~2 GB)
ollama pull llama3.2:3b
```

3. Ollama runs automatically in the background after installation. Verify:

```bash
ollama list          # shows downloaded models
ocr-kb models        # shows models + current config
```

---

## LM Studio setup (alternative)

1. Install [LM Studio](https://lmstudio.ai) and download at least one multimodal model.
2. Go to **Local Server** → enable **Multi-model mode**.
3. Load your vision model on port **1234** and text model on port **1235**.
4. Set `BACKEND_PROVIDER=lmstudio` in `local.env`.

---

## Configuration

```bash
cp local.env.example local.env
```

**Ollama (default):**

```bash
BACKEND_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_VISION_MODEL=llava:7b      # vision model for OCR and figure extraction
OLLAMA_TEXT_MODEL=gemma3:4b       # text model for enrichment and RAG
MODEL_BACKEND=hybrid              # hybrid | glm_only | gemma_only
```

**LM Studio:**

```bash
BACKEND_PROVIDER=lmstudio
GLM_OCR_BASE_URL=http://localhost:1234/v1
GLM_OCR_MODEL_NAME=gemma-4-it
GEMMA_BASE_URL=http://localhost:1235/v1
GEMMA_MODEL_NAME=gemma-4-it
MODEL_BACKEND=hybrid
```

**Storage and quality:**

```bash
KB_DB_PATH=./kb_data/kb.db
MARKDOWN_DIR=./kb_data/markdown
IMAGE_DPI=150                     # raise to 300 for dense/small text

# Embedded image extraction from PDFs
EXTRACT_EMBEDDED_IMAGES=true
EMBEDDED_IMAGE_MIN_PIXELS=10000   # skip images smaller than ~100×100

# VRAM limits (RTX 4070 12 GB defaults)
MAX_IMAGE_PIXELS=1310720          # hard cap before sending to vision model
MODEL_MAX_NEW_TOKENS=2048         # generation length cap
RAG_CHUNK_CHARS=600               # chars per retrieved snippet in ask
```

**Semantic search (enabled by default):**

```bash
ENABLE_EMBEDDINGS=true            # hybrid FTS5 + semantic search via Reciprocal Rank Fusion

# SPECTER2 (default) — asymmetric academic embeddings from AllenAI.
# Uses a different adapter for indexing documents vs. querying, giving
# better precision on paper retrieval. Requires pip install 'ocr-kb[specter2]'.
# Falls back to all-MiniLM-L6-v2 automatically if adapters is not installed.
EMBEDDING_MODEL=specter2

# Alternative: plain sentence-transformers model (HuggingFace, auto-cached):
# EMBEDDING_MODEL=all-MiniLM-L6-v2

# Alternative: Ollama-served embedding model (name:tag format auto-detected):
# EMBEDDING_MODEL=qwen3-embedding:4b
```

Ollama embedding models are routed through `OLLAMA_BASE_URL/v1/embeddings` automatically — no extra config needed. For sentence-transformers: `pip install sentence-transformers`. If neither is available, search degrades to FTS5-only.

**Metadata extraction:**

```bash
ENABLE_METADATA_EXTRACTION=true   # on by default; extracts title, authors, year,
                                  # DOI, abstract, journal from the first page
```

Metadata uses the free [Crossref API](https://api.crossref.org) (no key needed) when a DOI is detected, and falls back to LLM extraction otherwise. Extracted metadata is stored in the DB and exposed via `search --author/--year/--doi` filters and `show`.

**Vision model timeout:**

```bash
VISION_TIMEOUT=120                # seconds per Ollama/LM Studio OCR call
                                  # raise if figure-heavy pages time out (default 120)
```

**PDF extraction — pick a profile:**

Choose the block that matches your setup in `local.env` (see `local.env.example` for full details):

| Profile | Key settings | Best for |
|---------|-------------|----------|
| **Default — granite-docling on CUDA** | `USE_DOCLING=true` `DOCLING_DEVICE=cuda` | Most setups; automatic CPU fallback when VRAM is occupied |
| **granite-docling on CPU** | `USE_DOCLING=true` `DOCLING_DEVICE=cpu` | No GPU, or when you need all VRAM for other models |
| **pdftext + docling** | `USE_PDFTEXT=true` `USE_DOCLING=true` | Speed boost for digital PDFs: pdftext for clean pages, docling for the rest |
| **marker-pdf (CPU)** | `USE_DOCLING=false` `USE_MARKER_PDF=true` `MARKER_DEVICE=cpu` | Alternative; heavier but different layout model |

The extraction stack tries each enabled layer in order; the first success wins:

1. **pdftext** (`USE_PDFTEXT`) — reads the native PDF text layer with zero ML; instant for digital PDFs
2. **granite-docling** (`USE_DOCLING`, **default on**) — local VLM; handles both digital and scanned PDFs
3. **marker-pdf** (`USE_MARKER_PDF`) — layout-aware markdown via Surya models
4. **pymupdf4llm** (`USE_PYMUPDF4LLM`) — structured markdown from the native text layer
5. **pypdfium2** — plain text fallback; always available
6. **Surya OCR** (`USE_SURYA_OCR`) — for scanned pages when earlier layers produce blank output
7. **Vision model** — Ollama / LM Studio OCR as last resort

**Monitoring and error recovery:**

```bash
MAX_RETRIES=3                     # how many times to retry failed documents before giving up
```

Logs go to `kb_data/logs/ocr_kb.log` (rotating: 10 MB × 5 backups). Includes ingest speed, search latency, and error tracking. Use `ocr-kb dlq` commands to manage failed documents.

All settings can be overridden as environment variables.

---

## CLI quick-start

```bash
# List available models and current config
ocr-kb models

# Ingest a single PDF
ocr-kb ingest paper.pdf --tags "ml,2024" --category research

# Ingest a whole folder
ocr-kb ingest ./papers/ --category research

# Ask a question (RAG)
ocr-kb ask "What is backpropagation?"

# Search
ocr-kb search "gradient descent"

# Browse recent entries
ocr-kb recent
```

---

## CLI reference

### `models` — list available models

```bash
ocr-kb models
```

When `BACKEND_PROVIDER=ollama`, queries Ollama for all downloaded models, shows
their size, family, and whether they support vision input. Also shows the currently
configured vision and text models.

```
Ollama models at http://localhost:11434

  Model                               Size    Family    Type
  gemma3:4b                           3.3 GB  gemma3    TEXT
  llava:7b                            4.7 GB  llama     VISION
  qwen2.5vl:7b                        5.2 GB  qwen2     VISION

Current:  vision=llava:7b   text=gemma3:4b
```

---

### `ingest` — add documents

```bash
ocr-kb ingest <path> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--tags TEXT` | — | Comma-separated tags, e.g. `"ml,paper,2024"` |
| `--category TEXT` | — | Category label, e.g. `"biology"` |
| `--mode [text\|html]` | `text` | `html` preserves document structure |
| `--metadata/--no-metadata` | on | Extract title, authors, year, DOI, abstract, journal from first page |
| `--key-points` | off | Extra LLM pass to extract key points per page |
| `--summary` | off | Extra LLM pass to generate a one-paragraph summary |
| `--pages TEXT` | all | Page range, e.g. `"1,3-5"` — re-ingest specific pages only |
| `--vision-model TEXT` | from config | Override vision/OCR model for this run only |
| `--text-model TEXT` | from config | Override text/enrichment model for this run only |

Accepts: PDF, PNG, JPG, TIFF, plain text, Markdown. Directories are processed
recursively. Re-ingesting the same file updates existing entries in-place —
existing tags and category are **preserved** if not explicitly specified.

By default, pages are processed by **granite-docling** — a local VLM that renders
each page as an image and produces clean, structured markdown including headings,
tables, formulas, lists, and figure content. It handles both digital and scanned
PDFs with the same code path. No external server is required.

Optionally, with `EXTRACT_EMBEDDED_IMAGES=true` and an Ollama/LM Studio vision
model running, PDF-embedded figure objects are additionally captioned and appended
as `[Figure N]: ...`, making them individually searchable.

```bash
# Ingest with summaries and key points
ocr-kb ingest paper.pdf --summary --key-points

# Use a higher-quality vision model just for this PDF
ocr-kb ingest complex_diagram.pdf --vision-model qwen2.5vl:7b

# Ingest text-only files with a faster text model
ocr-kb ingest notes/ --category lecture --text-model llama3.2:3b

# HTML mode preserves document structure (useful for formatted PDFs)
ocr-kb ingest report.pdf --mode html
```

---

### `ask` — RAG question answering

```bash
ocr-kb ask <question> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--context N` | 5 | Number of documents to retrieve as context |
| `--chunk-chars N` | from config | Characters per retrieved snippet |
| `--text-model TEXT` | from config | Override text model for this answer only |

Returns a cited answer followed by a numbered source map:

```
Backpropagation is an algorithm for computing gradients... [1][3]

Sources:
  [1] deep_learning.pdf  p.4  id=12  [ml,fundamentals]
  [3] lecture_notes.pdf  p.9  id=31  [ml,2024]
```

```bash
ocr-kb ask "How does dropout prevent overfitting?" --context 8
ocr-kb ask "Summarise the key findings" --text-model gemma3:12b
ocr-kb ask "What activation functions are compared?" --chunk-chars 1200
```

---

### `search` — full-text search

```bash
ocr-kb search <query> [options]
```

| Option | Description |
|--------|-------------|
| `--limit N` | Maximum results (default 10) |
| `--source TEXT` | Restrict to entries whose source path contains this substring |
| `--category TEXT` | Exact category match |
| `--tag TEXT` | Tags contain this value |
| `--author/-a TEXT` | Filter by author name (partial match) |
| `--year/-y INT` | Filter by publication year |
| `--doi TEXT` | Exact DOI match |
| `--after YYYY-MM-DD` | Only entries ingested on or after this date |

```bash
ocr-kb search "attention mechanism"
ocr-kb search "loss function" --category lecture-notes --after 2024-01-01
ocr-kb search "Figure 3" --source "goodfellow"    # search figure descriptions
ocr-kb search "malnutrition" --author "Chen" --year 2025
ocr-kb search "transformer" --doi "10.1145/3534678.3539087"
```

Natural language queries work — question marks and special characters are
handled automatically. Note: FTS5 uses exact token matching (no stemming), so
`"transformer"` won't match `"transformers"`. Use the root form or a shared substring.

---

### `show` — display a single entry

```bash
ocr-kb show <id>
```

Prints full content: source path, page, category, tags, ingestion date, and extracted text
(including any figure descriptions). When metadata was extracted on ingest, also shows the
full bibliographic block — title, authors, year, journal, DOI, and abstract. Use the ID
shown in `search` or `recent` output. `--open` launches the source file.

---

### `recent` — browse latest entries

```bash
ocr-kb recent [--limit N]
```

Lists the most recently ingested entries, newest first. Default limit: 20.

---

### `tag` — edit entry metadata

```bash
ocr-kb tag <id> [--add TEXT] [--remove TEXT] [--category TEXT]
```

Modifies tags and category in-place — no re-ingestion needed.

```bash
ocr-kb tag 42 --add "reviewed,important" --remove "draft" --category research
```

---

### `retag` — bulk-edit tags by search query

```bash
ocr-kb retag --query <query> [--add TEXT] [--remove TEXT] [--limit N]
```

Applies tag changes to all entries matching a search query.

```bash
ocr-kb retag --query "deep_learning" --add "textbook"
ocr-kb retag --query "draft" --remove "draft" --limit 500
```

---

### `delete` — remove entries

```bash
ocr-kb delete --id <id>        # delete a single entry
ocr-kb delete --source <path>  # delete all pages from a source file
```

---

### `export` — export the knowledge base

```bash
ocr-kb export --out <path> [--format json|markdown|obsidian] [--filter-tag TEXT] [--filter-category TEXT]
```

| Format | Output |
|--------|--------|
| `json` | Single JSON file with all entries |
| `markdown` | One `.md` file per entry |
| `obsidian` | Obsidian-compatible notes with YAML front matter and wikilinks |

```bash
ocr-kb export --format json --out kb_backup.json
ocr-kb export --format obsidian --out ~/obsidian-vault/kb --filter-tag ml
ocr-kb export --format markdown --out ./exports --filter-category lecture-notes
```

---

### `compile-wiki` — synthesize a concept wiki

```bash
ocr-kb compile-wiki [--limit N] [--since YYYY-MM-DD]
```

Runs LLM concept extraction on every stored page, then synthesizes dedicated
concept pages (definition, evidence, metrics, limitations, cross-references).
Output is written to `MARKDOWN_DIR/concepts/`.

```bash
ocr-kb compile-wiki                        # full rebuild
ocr-kb compile-wiki --since 2024-06-14    # incremental: only new entries
ocr-kb compile-wiki --limit 50             # process at most 50 pages
```

---

### `lint-wiki` — check wiki health

```bash
ocr-kb lint-wiki
```

Reports broken internal links, orphan concept pages, and missing front matter.

---

### `watch` — ingest new files automatically

```bash
ocr-kb watch [--dir PATH] [--interval N]
```

Watches a directory for new files and ingests them automatically.
Default directory is `WATCH_DIR` from settings (`./data/inputs`).

---

### `dlq` — manage failed documents

```bash
ocr-kb dlq list                  # show unresolved failed documents
ocr-kb dlq retry                 # re-process retryable documents
ocr-kb dlq clear --yes           # mark all as resolved (bulk clear)
```

Documents that fail to save (due to embedding errors, DB issues, etc.) are automatically
pushed to a dead-letter queue. Each document tracks retry count and max retries.

| Command | Description |
|---------|-------------|
| `dlq list` | Show all unresolved items: source path, error message, retry count, and last failure time |
| `dlq retry` | Re-process all items where `retry_count < max_retries`; mark resolved on success |
| `dlq clear --yes` | Bulk mark all items as resolved without prompting |

Failed documents are never deleted — they remain in the queue indefinitely until manually resolved or retried.

```bash
ocr-kb ingest ./papers/ --category research      # some files may fail
ocr-kb dlq list                                   # see what failed
ocr-kb dlq retry                                  # retry failed files
```

---

## MCP server — agent integration

The MCP server exposes all 12 KB operations as tools that any MCP-aware coding
agent (Claude Code, Cursor, OpenCode) can call directly.

### Start the server

```bash
ocr-kb-mcp                      # stdio transport (Claude Code, Cursor, OpenCode)
ocr-kb-mcp --transport sse      # SSE transport (web-based agents)
```

### Configure Claude Code

Add to `~/.claude/claude_desktop_config.json` (or `~/.claude/mcp.json`):

```json
{
  "mcpServers": {
    "ocr-kb": {
      "command": "/path/to/.venv/bin/ocr-kb-mcp",
      "args": [],
      "env": {
        "KB_DB_PATH": "/path/to/kb_data/kb.db",
        "BACKEND_PROVIDER": "ollama",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "OLLAMA_VISION_MODEL": "llava:7b",
        "OLLAMA_TEXT_MODEL": "gemma3:4b"
      }
    }
  }
}
```

### Configure Cursor

Add to `.cursor/mcp.json` in your project root — same structure as above.

### Available tools

| Tool | What it does |
|------|-------------|
| `kb_ingest` | Add a file or directory; accepts `vision_model` / `text_model` overrides |
| `kb_search` | Full-text search with optional filters |
| `kb_ask` | RAG question-answering; accepts `text_model` override |
| `kb_show` | Retrieve one entry by ID |
| `kb_recent` | List most recently ingested entries |
| `kb_tag` | Edit tags/category on one entry |
| `kb_retag` | Bulk-edit tags across a search result set |
| `kb_delete` | Delete by ID or source path |
| `kb_export` | Export to JSON, Markdown, or Obsidian |
| `kb_stats` | Entry count and unique source count |
| `kb_compile_wiki` | Synthesize concept wiki from the KB |
| `kb_lint_wiki` | Check wiki for broken links and missing metadata |

See `mcp_config.example.json` for full config blocks for Ollama, LM Studio, and SSE mode.

---

## Typical research workflow

```bash
# 1. Ingest a batch of PDFs — granite-docling handles OCR locally, no server needed.
#    --metadata is on by default: extracts title, authors, year, DOI, abstract via Crossref.
ocr-kb ingest ~/papers/neuroscience/ --category neuroscience --tags "review,2025"

# 2. Optional: start Ollama to enable summaries, key points, and RAG answers
ollama list   # confirm models are available
ocr-kb models

# 3. Ingest with LLM enrichment (requires Ollama)
ocr-kb ingest ~/papers/neuroscience/ --summary --key-points

# 4. Browse what was ingested; check extracted metadata
ocr-kb recent --limit 30
ocr-kb show 12    # shows bibliographic metadata block + full page text

# 5. Search by content, author, year, or DOI
ocr-kb search "synaptic plasticity" --category neuroscience
ocr-kb search "LTP" --author "Bliss" --year 2024
ocr-kb search "" --doi "10.1038/s41586-024-00001-0"   # retrieve a specific paper

# 6. Ask questions — RAG over all ingested pages, with citations
ocr-kb ask "What mechanisms underlie long-term potentiation?"
ocr-kb ask "What does Figure 3 show?" --context 8

# 7. Use a better model for a specific query
ocr-kb ask "Summarise the key findings on BDNF" --text-model gemma3:12b

# 8. Organise entries
ocr-kb retag --query "hebbian" --add "theory,reviewed"

# 9. Build the wiki incrementally
ocr-kb compile-wiki --since 2025-01-01
ocr-kb lint-wiki

# 10. Export to Obsidian
ocr-kb export --format obsidian --out ~/obsidian-vault/phd-kb --filter-category neuroscience
```

---

## Automation

### One-shot batch ingest

The simplest automation: point `ingest` at a folder and it processes everything
recursively, skipping nothing.

```bash
ocr-kb ingest ~/papers/ \
  --category research \
  --tags "imported" \
  --summary \
  --vision-model llava:7b \
  --text-model gemma3:4b
```

Re-running the same command is safe — existing entries are updated in-place and
tags/category are preserved.

---

### Watch a folder for new files

`ocr-kb watch` monitors a directory and ingests any new file the moment it appears.
Drop a PDF into the folder and it's in the KB within seconds.

```bash
# Watch ~/papers/inbox, ingest everything that lands there
ocr-kb watch --dir ~/papers/inbox --category inbox

# Auto-compile the wiki after every new file
ocr-kb watch --dir ~/papers/inbox --auto-compile --auto-lint
```

The default watch directory is `WATCH_DIR` from `local.env` (`./data/inputs`).

---

### Run the watcher as a systemd service (Linux)

Create `/etc/systemd/system/ocr-kb-watch.service`:

```ini
[Unit]
Description=ocr-kb folder watcher
After=default.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/path/to/glm_ocr_kb
EnvironmentFile=/path/to/glm_ocr_kb/local.env
ExecStart=/path/to/glm_ocr_kb/.venv/bin/ocr-kb watch \
    --dir /home/YOUR_USERNAME/papers/inbox \
    --category inbox \
    --auto-compile
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
```

Enable and start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ocr-kb-watch

# Check logs
journalctl -u ocr-kb-watch -f
```

The watcher restarts automatically on failure (e.g. if Ollama wasn't ready yet).

---

### Error recovery — retry failed documents

When ingesting a batch, some documents may fail (e.g. corrupted PDF, temporary network hiccup,
or embedding model crash). Instead of losing progress, failed documents are pushed to a
**dead-letter queue** and can be retried later:

```bash
# Ingest a batch — some may fail
ocr-kb ingest ~/papers/ --category research --summary

# Check what failed
ocr-kb dlq list

# Retry everything that failed (up to MAX_RETRIES times)
ocr-kb dlq retry

# Or clear the queue without retrying
ocr-kb dlq clear --yes
```

Each document tracks the error message, retry count (0-based), and timestamp of the last failure.
Documents are retried up to `MAX_RETRIES` times before giving up (default: 3).

This is especially useful in production automation — if a batch ingest script encounters
issues, you can inspect the dead-letter queue, fix the underlying problem (e.g. upgrade Ollama),
and retry without re-ingesting the entire batch.

---

### Cron job — nightly batch ingest

If you prefer polling over watching (e.g. a network share that syncs overnight),
run a cron job that ingests a folder and rebuilds the wiki.

```bash
crontab -e
```

```cron
# Every night at 02:00 — ingest new papers and rebuild wiki
0 2 * * * cd /path/to/glm_ocr_kb && \
    .venv/bin/ocr-kb ingest ~/papers/ --category research --summary >> \
    ~/.local/share/ocr-kb/cron.log 2>&1 && \
    .venv/bin/ocr-kb compile-wiki --since "$(date -d yesterday +\%Y-\%m-\%d)" >> \
    ~/.local/share/ocr-kb/cron.log 2>&1
```

The `--since yesterday` flag makes `compile-wiki` only process entries ingested
in the last 24 hours, keeping the nightly run fast regardless of KB size.

---

### Shell script — ingest a folder tree with per-subfolder categories

If your papers are organised into topic folders, this script ingests each
subfolder with the folder name as the category:

```bash
#!/usr/bin/env bash
# ingest_all.sh — ingest ~/papers/<topic>/ with category=<topic>
set -euo pipefail

PAPERS_ROOT="${1:-$HOME/papers}"
VENV="/path/to/glm_ocr_kb/.venv/bin"

for topic_dir in "$PAPERS_ROOT"/*/; do
    category=$(basename "$topic_dir")
    echo "==> Ingesting $category ..."
    "$VENV/ocr-kb" ingest "$topic_dir" \
        --category "$category" \
        --summary \
        --tags "auto-imported"
done

echo "==> Rebuilding wiki ..."
"$VENV/ocr-kb" compile-wiki
"$VENV/ocr-kb" lint-wiki

echo "Done."
```

```bash
chmod +x ingest_all.sh
./ingest_all.sh ~/papers/
```

---

### Automation with the MCP server

When using Claude Code or Cursor, you can automate ingestion through natural
language — no shell scripts needed:

```
"Ingest everything in ~/papers/neurips2024/ with category 'neurips2024' and
tag 'conference'. Then rebuild the wiki and tell me how many new concepts were found."
```

The agent will call `kb_ingest`, `kb_compile_wiki`, and `kb_stats` in sequence.

---

## Project layout

```
ocr_kb/
  settings.py          pydantic-settings config (all env vars + routing properties)
  pipeline.py          main orchestration: file → pages → entries
  cli.py               Typer CLI (ocr-kb command)
  mcp_server.py        FastMCP server (ocr-kb-mcp command)
  monitoring.py        logging, timing, observability (ingest speed, search latency, error tracking)
  prompts.py           LLM prompt templates (OCR, figure description, RAG, wiki)
  ingest/
    loader.py              file-type detection and routing
    batch_builder.py       batching for multi-page docs; runs extraction priority chain per page
    pdf_reader.py          PDF → page images + native text (pypdfium2); strips headers/footers
    docling_reader.py      granite-docling VLM extraction (default); CUDA with auto CPU fallback;
                           strips preprint line numbers (medRxiv/bioRxiv sequential numbering)
    pdftext_reader.py      zero-ML native text extraction via pdftext (instant for digital PDFs)
    pymupdf4llm_reader.py  structured markdown extraction for native-text PDFs (pymupdf4llm)
    marker_reader.py       layout-aware markdown via full marker-pdf pipeline (5 Surya models)
    surya_reader.py        direct Surya OCR — detection + recognition only (2 models, lighter)
    image_reader.py        image loading and pre-processing
    metadata_extractor.py  bibliographic metadata: DOI→Crossref API, LLM fallback
  model/
    glm_ocr_backend.py vision model client — routes to Ollama or LM Studio (VISION_TIMEOUT aware)
    gemma_backend.py   text model client — routes to Ollama or LM Studio
    embedder.py        SPECTER2 (proximity/adhoc_query adapters) and sentence-transformers
  postprocess/
    html_parser.py     HTML cleaning (BeautifulSoup)
    markdown_converter.py HTML/plain → Markdown
    layout_parser.py   structural block extraction
  kb/
    schema.py          KBEntry dataclass
    store.py           SQLite upsert, CRUD, FTS5 triggers
    indexer.py         FTS5 search, filtered search
    exporter.py        JSON, Markdown, Obsidian export
    ingestion.py       shared ingest logic (CLI + MCP)
  wiki/
    compiler.py        concept extraction and wiki synthesis
    linter.py          wiki health checker
data/
  inputs/              drop files here for `ocr-kb watch`
kb_data/               runtime database and Markdown mirror
  kb.db                SQLite knowledge base
  markdown/            flat .md mirror of every KB entry
    concepts/          synthesized wiki concept pages
tests/                 444 unit tests (pytest, ~3 s, no real model calls)
local.env.example      configuration template
mcp_config.example.json  agent integration config blocks
```

---

## Running tests

```bash
pytest                        # all 444 tests
pytest tests/test_cli.py -v   # CLI only
pytest -k "search"            # filter by name
```

All tests mock the model layer — no Ollama or LM Studio instance required.

---

## Troubleshooting

**Ollama not reachable**
Run `ollama serve` to start the daemon, or check that it's running with `ollama list`.
Default port is 11434. Set `OLLAMA_BASE_URL` if you've changed it.

**Model not found (`model not found` error from Ollama)**
Pull the model first: `ollama pull llava:7b`. Use `ocr-kb models` to see what's
available and what's currently configured.

**Out of memory / CUDA OOM during ingest**
granite-docling automatically falls back to CPU when CUDA OOM is detected (e.g. Ollama
has a model loaded). You'll see a warning in the logs but the ingest will complete.
To force CPU from the start: `DOCLING_DEVICE=cpu`. For marker-pdf OOM, set
`MARKER_DEVICE=cpu`. To skip per-figure captioning, leave `EXTRACT_EMBEDDED_IMAGES=false`
(the default) or set `EMBEDDED_IMAGE_MIN_PIXELS=40000`.

**No results found when searching**
FTS5 uses exact token matching — `"transformer"` won't match `"transformers"`.
Use the exact token or a shared substring. Re-ingest with `--mode html` if the
original extraction was poor.

**Figure descriptions not appearing**
`EXTRACT_EMBEDDED_IMAGES` is `false` by default — granite-docling captures figure
content as part of its full-page OCR pass. If you want additional per-figure prose
captions, set `EXTRACT_EMBEDDED_IMAGES=true` and ensure an Ollama or LM Studio
vision model is running. Note this only works for PDFs with vector-embedded image
objects; scanned page bitmaps are already handled by the full-page OCR.

**MCP server not found by agent**
Use the absolute path to `ocr-kb-mcp`: `/full/path/to/.venv/bin/ocr-kb-mcp`.
Relative paths do not work in most agent MCP configs.

**LM Studio model name mismatch**
The model name must exactly match the identifier shown in LM Studio's server panel
(e.g. `gemma-4-it`, not `Gemma 4 IT`). Case-sensitive.

**Embeddings disabled / semantic search not working**
By default, `ENABLE_EMBEDDINGS=true` and `EMBEDDING_MODEL=specter2`. If you see adapter or embedding warnings:
```bash
# Install SPECTER2 support (recommended for academic papers):
pip install 'ocr-kb[specter2]'   # installs the 'adapters' library

# Or fall back to all-MiniLM-L6-v2 (no extra install, just change config):
# EMBEDDING_MODEL=all-MiniLM-L6-v2
```
To disable embeddings entirely (reduces memory, FTS5-only search):
```bash
ENABLE_EMBEDDINGS=false
```
If you switch embedding models after ingesting documents, re-ingest to regenerate vectors — old embeddings at a different dimension are skipped automatically.

**Vision model pages timing out**
If some pages fail with `Request timed out`, raise the timeout (default is 120 s):
```bash
VISION_TIMEOUT=180    # seconds per vision-model API call
```
This happens when the Ollama/LM Studio OCR fallback is called for figure-heavy pages that take longer than the default to process.

**Failed documents in dead-letter queue**
If ingestion fails partway through a batch, use `ocr-kb dlq` to inspect and retry:
```bash
ocr-kb dlq list        # see what failed
ocr-kb dlq retry       # retry failed documents
ocr-kb dlq clear --yes # or mark all as resolved without retry
```
Failed documents are tracked with error messages and retry count. Each document can be retried
up to `MAX_RETRIES` times (default: 3) before being considered permanently failed.

**Where are the logs?**
Structured logs are written to `kb_data/logs/ocr_kb.log` (rotating file, 10 MB × 5 backups).
This includes ingestion speed, search latency, and error tracking. Errors also appear in the CLI output.
