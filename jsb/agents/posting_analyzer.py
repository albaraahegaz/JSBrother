"""
JSBrother — Posting Analyzer Agent
Analyzes job/project postings → PostingJSON.
Must-have enrichment chain: LLM → stricter retry → rule-based fallback.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import config
from jsb.schemas import PostingJSON
from jsb.tools.web_fetcher import fetch_posting_or_text
from jsb.json_utils import extract_json
from jsb.llm_client import get_client
from jsb.skills import get_alias_registry, normalize_skill

logger = logging.getLogger("jsbrother.posting_analyzer")

MIN_MUST_HAVES = 2

EXTRACTION_SYSTEM_PROMPT = """You are an expert technical recruiter and job market analyst AI.
Your task is to analyze job postings and freelance project briefs, extracting all requirements into structured JSON.
Be precise and thorough in identifying technical and professional requirements:
- Distinguish between mandatory requirements (must_have) and optional/preferred ones (nice_to_have).
- When a requirement lists examples or alternatives (e.g., "experience with a deep learning framework like PyTorch or TensorFlow", "relational databases e.g. PostgreSQL, MySQL", "NLP / LLMs"), split and extract EACH specific technology as an individual item.
- Extract deal-breakers: strict disqualifiers or hard constraints (e.g., "Must be US citizen", "4+ years professional experience", "Required: 2+ years Python").
- For freelance projects, identify budget (amount, currency, kind: fixed/hourly), deadline (in days), and flag vague_scope=true if the brief is ambiguous, missing specifications, or lacks clear deliverables.
- Output ONLY a valid JSON object matching the required schema. No explanations, no markdown fences."""

STRICT_MUST_REMINDER = """
CRITICAL: Your previous extraction had too few must_have skills.
You MUST return at least 2 concrete must_have items (technologies, tools, or skills)
pulled from the Requirements / Qualifications / Must have sections.
Never return an empty must_have list. Output ONLY valid JSON.
"""

# Extra common terms not always covered as canonical aliases
_EXTRA_TERMS = [
    "python", "java", "scala", "rust", "sql", "nosql", "pandas", "numpy",
    "spark", "hadoop", "airflow", "kafka", "mlflow", "wandb", "opencv",
    "fastapi", "django", "flask", "streamlit", "gradio", "docker", "kubernetes",
    "aws", "gcp", "azure", "linux", "bash", "git", "ci/cd", "rest", "graphql",
    "pytorch", "tensorflow", "keras", "scikit-learn", "xgboost", "lightgbm",
    "transformers", "huggingface", "langchain", "langgraph", "faiss", "chroma",
    "pinecone", "weaviate", "openai", "llm", "rag", "nlp", "computer vision",
    "machine learning", "deep learning", "data science", "etl", "api",
]


def build_posting_prompt(posting_text: str) -> str:
    return f"""Please analyze the following job or project posting text and extract all details into a valid JSON object.

Required JSON Structure:
{{
  "title": "Job or Project Title",
  "company_or_client": "Company or Client Name",
  "type": "job",
  "must_have": ["python", "machine learning", "pytorch", "sql"],
  "nice_to_have": ["docker", "kubernetes"],
  "min_years": 3.0,
  "deal_breakers": [],
  "budget": null,
  "deadline_days": null,
  "vague_scope": false,
  "summary": "Concise 1-2 sentence summary.",
  "contact_email": null
}}

Guidelines:
1. Extract every technology and skill from requirements into must_have (at least 2).
2. Split "e.g." / "or" alternatives into separate items.
3. Keep skills lowercase and concise.

Posting Text:
\"\"\"
{posting_text}
\"\"\"

Output ONLY the JSON object."""


def _requirement_sections(text: str) -> str:
    """Prefer Requirements / Qualifications / Must-have sections when present."""
    lower = text.lower()
    markers = [
        "requirements", "qualifications", "must have", "must-have",
        "required skills", "what you'll need", "you should have",
        "skills required", "minimum qualifications",
    ]
    best_start = -1
    for m in markers:
        idx = lower.find(m)
        if idx != -1 and (best_start == -1 or idx < best_start):
            best_start = idx
    if best_start == -1:
        return text
    return text[best_start:]


def extract_skills_fallback(posting_text: str) -> Tuple[List[str], List[str]]:
    """
    Rule-based skill extractor using alias registry + common tech terms.
    Returns (must_have, nice_to_have).
    """
    registry = get_alias_registry()
    section = _requirement_sections(posting_text)
    hay = section.lower()

    # Build candidate phrases: all alias variants + extras, longest first
    phrases: List[str] = []
    for group in registry.canonical_to_group.values():
        phrases.extend(group)
    phrases.extend(_EXTRA_TERMS)
    # Unique, longest first to prefer "machine learning" over "ml"
    seen = set()
    uniq = []
    for p in phrases:
        n = normalize_skill(p)
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    uniq.sort(key=len, reverse=True)

    found: List[str] = []
    for phrase in uniq:
        # Word-boundary-ish match
        pat = r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])"
        if re.search(pat, hay):
            canonical = registry.get_canonical(phrase)
            if canonical not in found:
                found.append(canonical)
        if len(found) >= 12:
            break

    # Prefer first hits as must; rest as nice
    if len(found) >= 2:
        must = found[: max(2, min(8, len(found)))]
        nice = found[len(must):]
        return must, nice
    if len(found) == 1:
        return found, []
    # Last resort: scan bullet lines for capitalized tech tokens
    bullets = re.findall(r"[-•*]\s*([^\n]{3,80})", section)
    for b in bullets:
        for token in re.split(r"[,/;|]| and | or ", b):
            n = normalize_skill(token)
            if 2 <= len(n) <= 40 and n not in found:
                found.append(n)
            if len(found) >= 4:
                break
        if len(found) >= 4:
            break
    must = found[: max(2, len(found))] if found else ["python", "communication"]
    return must, found[len(must):]


def _set_extraction_path(posting: PostingJSON, path: str) -> PostingJSON:
    object.__setattr__(posting, "_extraction_path", path) if False else None
    # Pydantic may block setattr on model; use __dict__
    try:
        posting.__dict__["_extraction_path"] = path
    except Exception:
        setattr(posting, "_extraction_path", path)
    return posting


def get_extraction_path(posting: PostingJSON) -> str:
    return getattr(posting, "_extraction_path", "llm")


def analyze_posting(
    posting_source: Union[str, Path],
    save_path: Optional[Path] = None,
) -> PostingJSON:
    """
    Parses a job or project posting from text, local file, or URL.
    Ensures >= 2 must_haves via LLM → stricter retry → rule-based fallback.
    """
    posting_text = fetch_posting_or_text(posting_source)
    if not posting_text.strip():
        raise ValueError("Cannot analyze posting: Posting text is empty.")

    client = get_client()
    prompt = build_posting_prompt(posting_text)

    def _generate_retry(retry_prompt: str, system: Optional[str] = None) -> str:
        return client.generate(
            prompt=retry_prompt,
            system=system or EXTRACTION_SYSTEM_PROMPT,
            max_new_tokens=1000,
            temperature=config.TEMP_EXTRACTION,
        )

    raw_response = client.generate(
        prompt=prompt,
        system=EXTRACTION_SYSTEM_PROMPT,
        max_new_tokens=1000,
        temperature=config.TEMP_EXTRACTION,
    )

    posting = extract_json(
        raw_text=raw_response,
        schema_cls=PostingJSON,
        prompt_context=prompt,
        system_context=EXTRACTION_SYSTEM_PROMPT,
        max_retries=config.DEFAULT_MAX_RETRIES,
        generate_fn=_generate_retry,
    )
    path = "llm"

    if len(posting.must_have or []) < MIN_MUST_HAVES:
        logger.info(
            "Posting must_have count=%s < %s — stricter LLM retry",
            len(posting.must_have or []),
            MIN_MUST_HAVES,
        )
        retry_prompt = prompt + "\n" + STRICT_MUST_REMINDER
        raw2 = client.generate(
            prompt=retry_prompt,
            system=EXTRACTION_SYSTEM_PROMPT + "\nAlways include at least 2 must_have skills.",
            max_new_tokens=1000,
            temperature=config.TEMP_EXTRACTION,
        )
        try:
            posting2 = extract_json(
                raw_text=raw2,
                schema_cls=PostingJSON,
                prompt_context=retry_prompt,
                system_context=EXTRACTION_SYSTEM_PROMPT,
                max_retries=1,
                generate_fn=_generate_retry,
            )
            if len(posting2.must_have or []) >= MIN_MUST_HAVES:
                posting = posting2
                path = "retry"
            else:
                # Merge any skills found on retry
                merged = list(dict.fromkeys((posting.must_have or []) + (posting2.must_have or [])))
                posting.must_have = merged
                if len(merged) >= MIN_MUST_HAVES:
                    path = "retry"
        except Exception as e:
            logger.warning("Strict must_have retry failed: %s", e)

    if len(posting.must_have or []) < MIN_MUST_HAVES:
        logger.info("Using rule-based skill fallback for must_have")
        fb_must, fb_nice = extract_skills_fallback(posting_text)
        existing_must = list(posting.must_have or [])
        existing_nice = list(posting.nice_to_have or [])
        posting.must_have = list(dict.fromkeys(existing_must + fb_must))
        posting.nice_to_have = list(
            dict.fromkeys(
                [n for n in existing_nice + fb_nice if n not in posting.must_have]
            )
        )
        path = "fallback"

    _set_extraction_path(posting, path)
    logger.info(
        "Posting extraction_path=%s must_have=%s",
        path,
        posting.must_have[:8],
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(posting.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved extracted posting to %s", save_path)

    return posting
