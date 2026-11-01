"""
JSBrother — PDF Reader Tool
Extracts clean text from PDF documents using pypdf, with robust fallback
for raw pasted text and malformed inputs per §4 and §11 (M2).
"""

import os
from pathlib import Path
from typing import Union
import pypdf
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool


def extract_text_from_pdf(file_path: Union[str, Path]) -> str:
    """
    Reads a PDF file and extracts text across all pages.
    Raises FileNotFoundError if file does not exist, or ValueError if empty/corrupted.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found at: {path}")

    try:
        reader = pypdf.PdfReader(str(path))
        if len(reader.pages) == 0:
            raise ValueError(f"PDF file is empty (0 pages): {path}")

        extracted_pages = []
        for idx, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                # Normalize problematic unicode symbols that cause Windows cp1252 crashes
                normalized = (
                    page_text
                    .replace("\u2642", "")
                    .replace("\u2640", "")
                    .replace("\xa0", " ")
                    .replace("\u2018", "'")
                    .replace("\u2019", "'")
                    .replace("\u201c", '"')
                    .replace("\u201d", '"')
                    .replace("\u2022", "•")
                )
                extracted_pages.append(normalized.strip())

        full_text = "\n\n".join(extracted_pages).strip()
        if not full_text:
            raise ValueError(f"No extractable text found in PDF: {path}")

        return full_text
    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Failed to parse PDF {path}: {exc}") from exc


def read_pdf_or_text(file_path_or_text: str) -> str:
    """
    Primary entry point: extracts text from PDF if the input points to an existing file,
    otherwise treats the input as raw pasted CV/profile text fallback.
    """
    cleaned_input = (file_path_or_text or "").strip()
    if not cleaned_input:
        return ""

    # Check if the input is an existing file path ending in .pdf
    try:
        potential_path = Path(cleaned_input)
        if potential_path.is_file() and potential_path.suffix.lower() == ".pdf":
            return extract_text_from_pdf(potential_path)
    except Exception:
        pass

    # Fallback: treat as raw pasted text
    return cleaned_input


class PDFReaderInput(BaseModel):
    file_path_or_text: str = Field(
        description="Path to a PDF file or raw pasted text of a CV/profile."
    )


pdf_reader_tool = StructuredTool.from_function(
    func=read_pdf_or_text,
    name="pdf_reader",
    description="Extracts raw text from a CV PDF file, or processes raw pasted text.",
    args_schema=PDFReaderInput,
)
