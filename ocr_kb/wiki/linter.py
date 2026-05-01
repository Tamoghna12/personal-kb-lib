from __future__ import annotations

import re
from pathlib import Path

from ocr_kb.settings import Settings


def lint_wiki(settings: Settings) -> dict[str, list[str]]:
    """Check the wiki for broken links, orphan pages, and missing metadata."""
    issues = {
        "broken_links": [],
        "orphan_concepts": [],
        "missing_frontmatter": []
    }

    markdown_dir = settings.markdown_dir
    if not markdown_dir.exists():
        return {"error": ["Markdown directory does not exist."]}

    # 1. Collect all files
    all_files = list(markdown_dir.rglob("*.md"))
    file_set = {f.relative_to(markdown_dir) for f in all_files}
    
    # Concept names for orphan checking
    concept_links_found = set()

    for file_path in all_files:
        rel_path = file_path.relative_to(markdown_dir)
        content = file_path.read_text(encoding="utf-8")
        
        # Check Frontmatter
        if not content.startswith("---"):
            issues["missing_frontmatter"].append(str(rel_path))
        
        # Check Links [text](link)
        # Regex to find links, handling nested parentheses once (common in slugs with params)
        links = re.findall(r"\[.*?\]\(([^)]+(?:\([^)]*\)[^)]*)*)\)", content)
        for link in links:
            if link.startswith(("http", "mailto", "#")):
                continue
            
            # Resolve relative link
            link_path = (file_path.parent / link).resolve()
            try:
                link_rel = link_path.relative_to(markdown_dir.resolve())
                if link_rel not in file_set:
                    issues["broken_links"].append(f"{rel_path} -> {link}")
                
                if "concepts/" in str(link_rel):
                    concept_links_found.add(str(link_rel))
            except ValueError:
                issues["broken_links"].append(f"{rel_path} -> {link} (outside wiki)")

    # 2. Check for Orphan Concepts
    concepts_dir = markdown_dir / "concepts"
    if concepts_dir.exists():
        for concept_file in concepts_dir.glob("*.md"):
            rel_concept = concept_file.relative_to(markdown_dir)
            if str(rel_concept) not in concept_links_found:
                # Check index.md as it usually links everything
                index_content = (markdown_dir / "index.md").read_text() if (markdown_dir / "index.md").exists() else ""
                if str(rel_concept) not in index_content:
                    issues["orphan_concepts"].append(str(rel_concept))

    return issues
