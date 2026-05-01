"""Prompts for the GLM-OCR vision engine and Gemma 4 E4B enrichment layer.

All prompts are module-level constants so they can be imported and overridden
without touching pipeline logic.
"""

# ---------------------------------------------------------------------------
# GLM-OCR vision prompts  (sent alongside a page image)
# ---------------------------------------------------------------------------

GLM_OCR_PROMPT = (
    "Transcribe every word visible in this image exactly as written. "
    "Preserve headings, paragraphs, bullet lists, numbered lists, and tables. "
    "Maintain reading order top-to-bottom, left-to-right. "
    "Output plain text only — no commentary, no markdown formatting."
)

GLM_OCR_HTML_PROMPT = (
    "Convert all content visible in this image into a clean HTML fragment. "
    "Use <h1>–<h3> for headings, <p> for paragraphs, "
    "<ul>/<ol>/<li> for lists, and <table>/<tr>/<th>/<td> for tables. "
    "Preserve reading order. "
    "Output only the HTML fragment — no <html>, <head>, <body>, CSS, or JavaScript."
)

# Sent to the vision model for each embedded image (figure, diagram, chart)
# extracted from a PDF page. Goal: maximum information density for RAG retrieval.
IMAGE_FIGURE_PROMPT = (
    "Describe this image in full technical detail.\n"
    "• If it is a chart or graph: state the chart type, read every axis label, "
    "unit, and tick mark; describe each data series and its trend; state the "
    "key quantitative finding (e.g. 'accuracy reaches 94.3% at epoch 50').\n"
    "• If it is a diagram, flowchart, or architecture figure: name all components "
    "and describe every connection and direction of flow.\n"
    "• If it is a table rendered as an image: transcribe it exactly, row by row.\n"
    "• If it contains any text at all: transcribe every word verbatim.\n"
    "• If it is a photograph or illustration: describe the subject, context, "
    "and any visible labels or annotations.\n"
    "Be exhaustive and precise. Output plain text only — no preamble."
)

# ---------------------------------------------------------------------------
# Gemma enrichment prompts  (text-only, post-OCR)
# ---------------------------------------------------------------------------

# {text} is replaced at call time with the OCR'd or loaded content.

GEMMA_CLEANUP_PROMPT_TEMPLATE = (
    "Fix OCR errors, normalise spacing, and correct obvious transcription mistakes "
    "in the text below. Preserve all meaning and structure. Output only the corrected text.\n\n"
    "TEXT:\n{text}"
)

GEMMA_SUMMARY_PROMPT_TEMPLATE = (
    "Write a concise summary of the text below in 2–4 sentences. "
    "Focus on the main argument or findings. No preamble.\n\n"
    "TEXT:\n{text}"
)

GEMMA_TAGS_PROMPT_TEMPLATE = (
    "Generate a comma-separated list of 3–8 concise tags (lowercase, no spaces) "
    "that best describe the topics in the text below. Output only the tag list, nothing else.\n\n"
    "TEXT:\n{text}"
)

GEMMA_ENTITIES_PROMPT_TEMPLATE = (
    "Extract named entities from the text below as a JSON object with keys: "
    '"people", "organisations", "locations", "dates". '
    "Each key maps to a list of strings. Output valid JSON only.\n\n"
    "TEXT:\n{text}"
)

# Legacy key-points template (kept for pipeline compatibility)
_KEY_POINTS_TEMPLATE = (
    "Extract the most important ideas from the text below as a Markdown list.\n"
    "One bullet per idea. Maximum 10 bullets. No preamble or explanation.\n\n"
    "TEXT:\n{text}"
)


# ---------------------------------------------------------------------------
# Wiki synthesis prompts (Karpathy style)
# ---------------------------------------------------------------------------

WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE = (
    "Analyze the following technical text with extreme academic rigour and skepticism. "
    "Extract key scientific concepts, methodologies, experimental findings, and specific theories. "
    "For each item, extract the following fields strictly based on the text: "
    "1. 'definition': A precise technical definition or its specific role. "
    "2. 'evidence': Identify specific evidence or results. "
    "3. 'metrics': Extract quantitative data (e.g., sample size N, p-values, confidence intervals, effect sizes, accuracy %). "
    "4. 'limitations': Identify any stated boundary conditions, caveats, or limitations of the finding/methodology. "
    "5. 'citations': Extract formal academic citations (e.g., '[Smith et al., 2021]' or DOIs) associated with this item as a list of strings. "
    "Format the output as a JSON list of objects: "
    '[ {{"name": "Concept Name", "type": "methodology|theory|finding|entity", "definition": "...", "evidence": "...", "metrics": "...", "limitations": "...", "citations": ["..."]}} ]. '
    "Output valid JSON only.\n\n"
    "TEXT:\n{text}"
)

WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE = (
    "You are a senior scientific editor maintaining a rigorous academic knowledge base. "
    "Synthesize a comprehensive, highly critical summary for the concept: '{concept}'. "
    "Use the provided source snippets which include definitions, evidence, quantitative metrics, limitations, and citations. "
    "Structure the summary with the following sections EXACTLY: "
    "### Technical Overview\n"
    "### Methodology & Implementation\n"
    "### Quantitative Evidence & Results\n"
    "### Limitations & Caveats\n"
    "### Consensus & Contradictions\n"
    "(Analyze the snippets for conflicts. Explicitly state where sources agree and where their findings or methodologies conflict or diverge.)\n"
    "### Related Research & Citations\n\n"
    "Maintain a formal, objective, skeptical, and precise academic tone. Use Markdown. No preamble.\n\n"
    "SOURCE SNIPPETS:\n{snippets}"
)


# ---------------------------------------------------------------------------
# Formatter helpers
# ---------------------------------------------------------------------------

def format_glm_ocr_prompt(mode: str = "text") -> str:
    return GLM_OCR_HTML_PROMPT if mode == "html" else GLM_OCR_PROMPT


def format_key_points_prompt(text: str) -> str:
    return _KEY_POINTS_TEMPLATE.format(text=text)


def format_summary_prompt(text: str) -> str:
    return GEMMA_SUMMARY_PROMPT_TEMPLATE.format(text=text)


def format_cleanup_prompt(text: str) -> str:
    return GEMMA_CLEANUP_PROMPT_TEMPLATE.format(text=text)


def format_tags_prompt(text: str) -> str:
    return GEMMA_TAGS_PROMPT_TEMPLATE.format(text=text)


def format_entities_prompt(text: str) -> str:
    return GEMMA_ENTITIES_PROMPT_TEMPLATE.format(text=text)


def format_wiki_extraction_prompt(text: str) -> str:
    return WIKI_CONCEPT_EXTRACTION_PROMPT_TEMPLATE.format(text=text)


def format_wiki_concept_summary_prompt(concept: str, snippets: str) -> str:
    return WIKI_CONCEPT_SUMMARY_PROMPT_TEMPLATE.format(concept=concept, snippets=snippets)


# ---------------------------------------------------------------------------
# Metadata extraction prompt  (bibliographic info from first page of academic PDF)
# ---------------------------------------------------------------------------

METADATA_EXTRACTION_PROMPT_TEMPLATE = (
    "Extract bibliographic metadata from this academic paper's first page. "
    "Return ONLY valid JSON with exactly these fields "
    "(empty string \"\" if unknown, null for year if unknown):\n\n"
    '{{"title": "...", "authors": "First Last, First Last", '
    '"year": 2024, "doi": "10.xxxx/...", '
    '"abstract": "...", "journal": "..."}}\n\n'
    "First page text:\n---\n{text}\n---"
)


def format_metadata_prompt(text: str) -> str:
    return METADATA_EXTRACTION_PROMPT_TEMPLATE.format(text=text[:3000])


# ---------------------------------------------------------------------------
# RAG query prompt  (question-answering over retrieved KB context)
# ---------------------------------------------------------------------------

GEMMA_RAG_PROMPT_TEMPLATE = (
    "You are a precise, factual assistant with access to a personal knowledge base. "
    "Answer the question using ONLY the provided context. "
    "Cite source numbers inline (e.g. [1], [2]). "
    "If the context is insufficient, say so clearly — do not speculate.\n\n"
    "CONTEXT:\n{context}\n\n"
    "QUESTION:\n{question}"
)


def format_rag_prompt(question: str, context: str) -> str:
    return GEMMA_RAG_PROMPT_TEMPLATE.format(question=question, context=context)


# ---------------------------------------------------------------------------
# Backward-compatible aliases  (old names still importable)
# ---------------------------------------------------------------------------

OCR_PROMPT = GLM_OCR_PROMPT
HTML_EXTRACTION_PROMPT = GLM_OCR_HTML_PROMPT
