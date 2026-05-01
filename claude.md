 
# CLAUDE.md

You are working on a local OCR knowledge-base pipeline project.

## Goal
Build a reliable local OCR ingestion system that:
- accepts PDFs, images, and handwritten notes
- sends page images to a local multimodal model through LM Studio
- stores OCR output as Markdown, HTML, JSON, and SQLite records
- supports exact and semantic search
- stays local-first and easy to maintain

## Tech stack
- Python 3.11
- LM Studio OpenAI-compatible local API
- `openai` Python client or `claude` or something else
- `pypdfium2`
- `Pillow`
- `pydantic-settings`
- `beautifulsoup4`
- `markdownify`
- `sqlite-utils`
- `sentence-transformers`
- `typer`
- `rich`
- `watchdog`
- `pytest`

## Working rules
1. Make small, reversible changes.
2. Edit only the files relevant to the current task.
3. Write or update tests for every meaningful change.
4. Prefer explicit, readable Python over clever abstractions.
5. Keep OCR client logic isolated from file I/O and storage.
6. Keep the pipeline deterministic and local.
7. Do not refactor unrelated modules unless required.
8. If a task can be broken into smaller steps, break it down.
9. Summarize what changed and what remains after each step.
10. Stop and ask before changing architecture.

## Project architecture
- `ocr_kb/settings.py` for config
- `ocr_kb/ingest/` for image/PDF loading
- `ocr_kb/model/` for LM Studio OCR client
- `ocr_kb/postprocess/` for HTML/Markdown/layout parsing
- `ocr_kb/kb/` for SQLite storage and search
- `ocr_kb/pipeline.py` for orchestration
- `ocr_kb/cli.py` for user-facing commands

## Definition of done
A task is done only when:
- code is implemented
- tests are added or updated
- imports are clean
- behavior matches the requested scope
- any obvious edge cases are handled

## Prompting style for me
When giving instructions, be concrete:
- name the files to edit
- define the expected inputs and outputs
- specify what tests to add
- avoid vague or multi-goal requests

## Backend note
LM Studio is serving a local OpenAI-compatible API. Use the configured local base URL and model name from environment settings.