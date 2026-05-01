from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from ocr_kb.kb.store import get_all, init_db
from ocr_kb.model import run_enrichment
from ocr_kb.prompts import format_wiki_concept_summary_prompt, format_wiki_extraction_prompt
from ocr_kb.settings import Settings


def _slug(text: str) -> str:
    """Standardize concept names into filesystem-friendly slugs."""
    import re
    # Lowercase, replace spaces/slashes with dashes, remove other non-alphanumeric
    s = text.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s-]+", "-", s)
    return s[:80]


def compile_wiki(
    settings: Settings,
    limit: int | None = None,
    since: str | None = None,
) -> dict[str, int]:
    """Batch-process database entries to synthesize a wiki knowledge graph.

    *since* — ISO-8601 date/datetime string; only entries with created_at >=
    this value are processed.  Useful for incremental re-runs after new ingests
    without reprocessing the entire library.
    """
    conn = init_db(settings.kb_db_path)
    entries = get_all(conn)

    if since:
        entries = [e for e in entries if e.created_at.isoformat() >= since]
    if limit:
        entries = entries[:limit]

    # Concept -> list of (source_filename, context_snippet)
    concept_map: dict[str, list[dict]] = defaultdict(list)
    # Source -> list of found concepts
    source_to_concepts: dict[str, set[str]] = defaultdict(set)
    # Citation -> list of sources citing it
    citation_map: dict[str, set[str]] = defaultdict(set)

    # 1. Extraction Pass: Logical Chunking
    print(f"Extraction Pass: Processing {len(entries)} pages (grouped by section)...")
    
    current_section_title = "Introductory Content"
    current_section_text = []
    
    def process_chunk(title, text_list, source_name):
        full_text = "\n".join(text_list)
        if not full_text.strip(): return
        
        raw_concepts = run_enrichment(format_wiki_extraction_prompt(full_text), settings)
        try:
            clean_json = raw_concepts.strip()
            if clean_json.startswith("```json"): clean_json = clean_json[7:]
            if clean_json.endswith("```"): clean_json = clean_json[:-3]
            
            items = json.loads(clean_json.strip())
            for item in items:
                name = item["name"].strip()
                if not name: continue
                
                citations = item.get("citations", [])
                if isinstance(citations, str):
                    citations = [c.strip() for c in citations.split(",") if c.strip()]
                elif not isinstance(citations, list):
                    citations = []

                # Academic payload
                payload = {
                    "source": source_name,
                    "section": title,
                    "definition": item.get("definition", ""),
                    "evidence": item.get("evidence", ""),
                    "metrics": item.get("metrics", ""),
                    "limitations": item.get("limitations", ""),
                    "citations": citations,
                    "type": item.get("type", "concept")
                }
                concept_map[name].append(payload)
                source_to_concepts[source_name].add(name)
                
                for cit in citations:
                    if cit: citation_map[cit].add(source_name)
        except Exception as e:
            print(f"    ⚠ Extraction error in '{title}': {e}")

    source_name = "unknown"
    for i, entry in enumerate(entries):
        stem = Path(entry.source_path).stem
        source_name = f"{stem}_p{entry.page_number:03d}.md"
        
        # Parse blocks to find logical headings
        try:
            blocks = json.loads(entry.layout_blocks)
            for b in blocks:
                if b["type"] == "heading":
                    # Before changing section, process current chunk
                    if current_section_text:
                        process_chunk(current_section_title, current_section_text, source_name)
                        current_section_text = []
                    current_section_title = b["content"]
                else:
                    current_section_text.append(b["content"])
        except:
            # Fallback for pages without blocks
            current_section_text.append(entry.raw_text)

    # Final chunk
    if current_section_text:
        process_chunk(current_section_title, current_section_text, source_name)

    # 2. Synthesis Pass (Concepts)
    print(f"Synthesis Pass: Generating {len(concept_map)} academic concept pages...")
    concepts_dir = settings.markdown_dir / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (concept, mentions) in enumerate(concept_map.items()):
        print(f"  [{i+1}/{len(concept_map)}] Synthesizing '{concept}'...")
        _save_concept_page(concept, mentions, settings)

    # 2.5 Synthesis Pass (Citations)
    if citation_map:
        print(f"Synthesis Pass: Generating {len(citation_map)} citation stubs...")
        for citation, citing_sources in citation_map.items():
            _save_citation_stub(citation, list(citing_sources), settings)

    # 3. Catalog Pass (Index & Log)
    _update_index(settings, concept_map, source_to_concepts, list(citation_map.keys()))
    _write_log(settings, f"Compiled {len(entries)} pages. Synthesized {len(concept_map)} concepts and {len(citation_map)} citations.")

    return {
        "processed_entries": len(entries),
        "concepts_found": len(concept_map)
    }

def _save_citation_stub(citation: str, sources: list[str], settings: Settings):
    """Generate a simple stub page for a formal citation to track links."""
    slug = _slug(citation)
    path = settings.markdown_dir / "concepts" / f"{slug}.md"
    
    # Only create if it doesn't already exist (e.g., if it was synthesized as a full concept)
    if path.exists():
        return
        
    front_matter = [
        "---",
        f'title: "{citation}"',
        'type: citation',
        "sources:"
    ]
    for s in sorted(sources):
        front_matter.append(f'  - "{s}"')
    
    front_matter += [
        f"created: {datetime.now().date().isoformat()}",
        f"updated: {datetime.now().date().isoformat()}",
        "---",
        "",
        f"# {citation}",
        "",
        "This is an automatically generated citation stub. The following source pages reference this citation:",
        ""
    ]
    for s in sorted(sources):
        front_matter.append(f"- [{s}](../{s})")
        
    path.write_text("\n".join(front_matter), encoding="utf-8")

def _save_concept_page(concept: str, mentions: list[dict], settings: Settings):
    """Generate or update a dedicated concept page."""
    slug = _slug(concept)
    path = settings.markdown_dir / "concepts" / f"{slug}.md"
    
    sources_list = sorted(list(set(m["source"] for m in mentions)))
    # Build snippets with academic context
    snippets_list = []
    for m in mentions:
        snippet = f"- In section '{m['section']}' of {m['source']}:\n"
        snippet += f"  - Definition: {m['definition']}\n"
        if m.get('evidence'): snippet += f"  - Evidence: {m['evidence']}\n"
        if m.get('metrics'): snippet += f"  - Metrics: {m['metrics']}\n"
        if m.get('limitations'): snippet += f"  - Limitations: {m['limitations']}\n"
        if m.get('citations'):
            cits = ", ".join(m['citations']) if isinstance(m['citations'], list) else str(m['citations'])
            snippet += f"  - Citations: {cits}\n"
        snippets_list.append(snippet)
    
    snippets = "\n\n".join(snippets_list)
    
    # Generate synthesis via LLM
    summary = run_enrichment(format_wiki_concept_summary_prompt(concept, snippets), settings)
    
    front_matter = [
        "---",
        f'title: "{concept}"',
        f'type: {mentions[0]["type"]}',
        "sources:",
    ]
    for s in sources_list:
        front_matter.append(f'  - "{s}"')
    
    front_matter += [
        f"created: {datetime.now().date().isoformat()}",
        f"updated: {datetime.now().date().isoformat()}",
        "---",
        "",
        f"# {concept}",
        "",
        summary, # The LLM summary already contains headers based on our new prompt
        "",
        "## Mentions & Evidence",
        ""
    ]
    
    for m in mentions:
        front_matter.append(f"- [{m['source']}](../{m['source']}) (Section: {m['section']})")
        if m.get('definition'): front_matter.append(f"  - **Context:** {m['definition']}")
        if m.get('evidence'): front_matter.append(f"  - **Evidence:** {m['evidence']}")
        if m.get('metrics'): front_matter.append(f"  - **Metrics:** {m['metrics']}")
        if m.get('limitations'): front_matter.append(f"  - **Limitations:** {m['limitations']}")
        if m.get('citations'):
            cits = ", ".join(m['citations']) if isinstance(m['citations'], list) else str(m['citations'])
            front_matter.append(f"  - **Citations:** {cits}")
    
    path.write_text("\n".join(front_matter), encoding="utf-8")


def _update_index(settings: Settings, concepts: dict, sources: dict, citations: list[str] = None):
    """Regenerate the wiki index.md."""
    path = settings.markdown_dir / "index.md"
    lines = [
        "---",
        'title: "Wiki Index"',
        "type: index",
        f"created: {datetime.now().date().isoformat()}",
        "---",
        "",
        "# LLM Wiki Index",
        "",
        "## Concepts",
        ""
    ]
    for c in sorted(concepts.keys()):
        slug = _slug(c)
        lines.append(f"- [{c}](concepts/{slug}.md)")
        
    if citations:
        lines += ["", "## Citations", ""]
        for c in sorted(citations):
            slug = _slug(c)
            lines.append(f"- [{c}](concepts/{slug}.md)")
            
    lines += ["", "## Sources", ""]
    for s in sorted(sources.keys()):
        lines.append(f"- [{s}]({s})")
        
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_log(settings: Settings, message: str):
    """Append to log.md."""
    path = settings.markdown_dir / "log.md"
    
    if not path.exists():
        header = [
            "---",
            'title: "Wiki Log"',
            "type: log",
            "---",
            "",
            "# Wiki Activity Log",
            "",
            ""
        ]
        path.write_text("\n".join(header), encoding="utf-8")
    
    entry = f"- **{datetime.now().isoformat()}**: {message}\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)
