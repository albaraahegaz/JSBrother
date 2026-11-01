"""
JSBrother — Profile Analyzer Agent
Extracts structured ProfileJSON from CV (PDF or pasted text) using the LLM,
validates with json_utils, and persists to data/profile.json per §4 and §11 (M2).
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union, Dict, Any

import config
from jsb.schemas import ProfileJSON
from jsb.tools.pdf_reader import read_pdf_or_text
from jsb.json_utils import extract_json
from jsb.llm_client import generate, get_client

logger = logging.getLogger("jsbrother.profile_analyzer")

EXTRACTION_SYSTEM_PROMPT = """You are an expert technical recruiter and resume parsing AI.
Your task is to analyze candidate resumes/CVs and extract their full professional profile into clean, structured JSON.
Be exhaustive and precise when extracting skills: capture all programming languages, frameworks, libraries, tools, database technologies, machine learning/AI techniques, and methodologies mentioned across skills, experience, and project sections.
Output ONLY a valid JSON object matching the required schema. No explanations, no markdown fences."""


def build_profile_prompt(cv_text: str) -> str:
    return f"""Please extract the candidate's professional profile from the following resume/CV text into a valid JSON object.

Required JSON Structure:
{{
  "name": "Candidate Full Name",
  "email": "candidate@email.com",
  "headline": "Current or recent professional headline/title",
  "skills": ["python", "sql", "machine learning", "pytorch", ...],
  "years_experience": 2.5,
  "roles": [
    {{
      "title": "Role Title",
      "company": "Company Name",
      "years": 1.5,
      "highlights": ["Key achievement or responsibility 1", "Key achievement 2"]
    }}
  ],
  "education": ["Degree, Field of Study, Institution, Year"],
  "projects": ["Project Name: Brief summary of technologies and outcomes"],
  "languages": ["English", "Arabic"],
  "links": ["https://linkedin.com/in/...", "https://github.com/..."],
  "min_budget": null,
  "min_lead_days": 7
}}

Resume/CV Text:
\"\"\"
{cv_text}
\"\"\"

Output ONLY the JSON object."""


def analyze_profile(
    cv_source: Union[str, Path],
    min_budget: Optional[float] = None,
    min_lead_days: int = 7,
    save_to_disk: bool = True,
    profile_path: Optional[Path] = None,
) -> ProfileJSON:
    """
    Parses a CV from a file path or raw text, invokes the LLM for structured extraction,
    validates the output against ProfileJSON, and optionally persists to data/profile.json.
    """
    # 1. Read PDF or raw text fallback
    cv_text = read_pdf_or_text(str(cv_source))
    if not cv_text.strip():
        raise ValueError("Cannot analyze profile: CV text is empty.")

    # 2. Call LLM to extract profile
    client = get_client()
    prompt = build_profile_prompt(cv_text)

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

    # 3. Validate JSON with retry fallback
    profile = extract_json(
        raw_text=raw_response,
        schema_cls=ProfileJSON,
        prompt_context=prompt,
        system_context=EXTRACTION_SYSTEM_PROMPT,
        max_retries=config.DEFAULT_MAX_RETRIES,
        generate_fn=_generate_retry,
    )

    # 4. Apply optional user settings if provided
    if min_budget is not None:
        profile.min_budget = min_budget
    if min_lead_days is not None:
        profile.min_lead_days = min_lead_days

    # 5. Persist to data/profile.json
    if save_to_disk:
        target_file = profile_path or config.PROFILE_PATH
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text(
            json.dumps(profile.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Saved candidate profile to {target_file}")

    return profile


def load_saved_profile(profile_path: Optional[Path] = None) -> Optional[ProfileJSON]:
    """Loads previously persisted ProfileJSON if available."""
    target_file = profile_path or config.PROFILE_PATH
    if not target_file.exists():
        return None
    try:
        data = json.loads(target_file.read_text(encoding="utf-8"))
        return ProfileJSON.model_validate(data)
    except Exception as exc:
        logger.warning(f"Failed to load saved profile from {target_file}: {exc}")
        return None
