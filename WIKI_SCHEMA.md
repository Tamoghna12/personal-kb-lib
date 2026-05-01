# LLM Wiki Schema

This document defines the structural conventions for the `ocr-kb` LLM Wiki. The LLM (Gemma-4-E4B) acts as a disciplined wiki maintainer by adhering to these rules during the `compile-wiki` and `lint-wiki` operations.

## 1. Directory Structure

All wiki content resides in `kb_data/markdown/`:
- `*.md`: Raw source pages (one per document page).
- `concepts/*.md`: Knowledge-base articles synthesized from one or more sources.
- `index.md`: Master catalog of all sources and concepts.
- `log.md`: Append-only activity log of compiler actions.

## 2. Page Conventions

Every wiki page (sources and concepts) MUST have YAML frontmatter at the very top.

### Concept Page Frontmatter
```yaml
---
title: "Concept Name"
type: concept | entity | comparison
sources:
  - "source_file_p001.md"
related:
  - "Another Concept"
created: YYYY-MM-DD
updated: YYYY-MM-DD
---
```

### Source Page Frontmatter
```yaml
---
title: "Source Filename — Page N"
type: source-summary
source_path: "data/inputs/original.pdf"
page: N
created: YYYY-MM-DD
---
```

## 3. Linking Protocol

- **Backlinks:** Concept pages must link back to the specific source pages they derive from using relative paths: `[Source Title](../source_file_p001.md)`.
- **Cross-links:** Concepts should link to other related concepts: `[Related Concept](another-concept.md)`.
- **Automatic:** The Wiki Compiler identifies potential links based on concept names found in the text.

## 4. Maintenance Workflows

### Ingest
- Raw OCR text is converted to Markdown source pages.
- Source pages are stored in the root of the wiki directory.

### Compile
- LLM scans new source pages to extract entities and concepts.
- LLM updates or creates files in `concepts/`.
- LLM updates `index.md` and `log.md`.

### Lint
- Linter checks for missing frontmatter.
- Linter identifies broken relative links.
- Linter identifies "orphaned" concepts with no inbound links or no sources.
