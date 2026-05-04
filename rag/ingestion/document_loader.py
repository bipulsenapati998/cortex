"""
Layer 1 — Document loading from .txt, .md, .pdf, .docx files.
Layer 2 — Semantic chunking (paragraph-based with size cap).
"""

import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger("cortex.ingestion")
# Mapping of filename patterns to (doc_id prefix, access_tier, metadata category)
DOC_METADATA_RULES = {
    "hr_": ("hr", "internal", "hr_policy"),
    "parental": ("hr", "internal", "hr_policy"),
    "benefits": ("hr", "internal", "hr_policy"),
    "pto": ("hr", "internal", "hr_policy"),
    "it_": ("it", "internal", "technical"),
    "vpn": ("it", "internal", "technical"),
    "onboard": ("it", "internal", "technical"),
    "finance": ("finance", "confidential", "financial"),
    "board": ("exec", "secret", "board"),
    "faq": ("faq", "public", "product_faq"),
    "policy": ("policy", "internal", "hr_policy"),
    "compliance": ("legal", "confidential", "financial"),
    "runbook": ("it", "internal", "technical"),
    "ticket": ("support", "internal", "support_ticket"),
}


def _infer_metadata(filename: str) -> Dict:
    """Infer doc_id prefix, access_tier and category from filename."""
    fname = filename.lower()
    for pattern, (prefix, tier, category) in DOC_METADATA_RULES.items():
        if pattern in fname:
            stem = Path(filename).stem.replace(" ", "_")
            return {
                "doc_id": f"{prefix}_{stem}",
                "access_tier": tier,
                "category": category,
                "source": filename,
            }
    # Default: public FAQ
    stem = Path(filename).stem.replace(" ", "_")
    return {
        "doc_id": f"doc_{stem}",
        "access_tier": "public",
        "category": "product_faq",
        "source": filename,
    }


def load_text_file(filepath: str) -> str:
    """Load content from .txt file."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def paragraph_chunk(text: str, max_chars: int = 800) -> List[str]:
    """
    Layer 2: Split text into chunks at paragraph boundaries.
    Keeps chunks under max_chars while preserving paragraph coherence.
    """
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) < max_chars:
            current += para + "\n\n"
        else:
            if current:
                chunks.append(current.strip())
            # If single paragraph > max_chars, split by sentence
            if len(para) > max_chars:
                sentences = para.replace(". ", ".\n").split("\n")
                sub = ""
                for sentence in sentences:
                    if len(sub) + len(sentence) < max_chars:
                        sub += sentence + " "
                    else:
                        if sub:
                            chunks.append(sub.strip())
                        sub = sentence + " "
                if sub:
                    chunks.append(sub.strip())
                current = ""
            else:
                current = para + "\n\n"

    if current:
        chunks.append(current.strip())

    return [c for c in chunks if len(c) > 50]  # filter trivially short chunks
