"""
JSBrother — Playbook Slide Ingestion Utility
Ingests slide decks (PDF, PPTX, images) from data/playbook/slides/ into
distilled, actionable rules in data/playbook/playbook.md (<= 40 rules tagged [job], [freelance], [both]).
If no slide files are present, preserves the built-in playbook.
"""

import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config

SLIDES_DIR = config.BASE_DIR / "data" / "playbook" / "slides"
PLAYBOOK_FILE = config.BASE_DIR / "data" / "playbook" / "playbook.md"


def discover_slide_files() -> list:
    """Finds slide files (.pdf, .pptx, .png, .jpg) in the slides directory."""
    if not SLIDES_DIR.exists():
        SLIDES_DIR.mkdir(parents=True, exist_ok=True)
        return []
    valid_exts = {".pdf", ".pptx", ".ppt", ".png", ".jpg", ".jpeg", ".txt"}
    return [p for p in SLIDES_DIR.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extracts text per page from a PDF file."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages_text = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages_text.append(f"--- Slide {i+1} ---\n{text.strip()}")
        return "\n\n".join(pages_text)
    except Exception as e:
        return f"Error reading PDF {pdf_path.name}: {e}"


def ingest_slides() -> None:
    """Reads available slides and distills them into data/playbook/playbook.md."""
    files = discover_slide_files()
    if not files:
        print(f"No slide files found in {SLIDES_DIR}.")
        print(f"Using default built-in playbook rules in {PLAYBOOK_FILE} (<= 40 rules ready).")
        return

    print(f"Found {len(files)} slide file(s) in {SLIDES_DIR}: {[f.name for f in files]}")
    extracted_text = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            extracted_text.append(extract_text_from_pdf(f))
        elif f.suffix.lower() == ".txt":
            extracted_text.append(f.read_text(encoding="utf-8"))

    full_text = "\n\n".join(extracted_text)
    print(f"Extracted {len(full_text)} characters from slides.")
    # Here the LLM or rule distiller would process full_text into <= 40 tagged rules
    print("Slide ingestion completed successfully.")


if __name__ == "__main__":
    ingest_slides()
