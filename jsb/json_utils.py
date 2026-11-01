"""
JSBrother — JSON Utilities
Robust JSON extraction from LLM outputs: strips markdown code fences,
extracts outermost JSON object, validates against Pydantic schemas,
and retries up to 2 times on malformed outputs per §5.
"""

import json
import logging
import re
from typing import Type, TypeVar, Optional, Callable, Dict, Any, Tuple
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("jsbrother.json_utils")

T = TypeVar("T", bound=BaseModel)


class JSONExtractionError(ValueError):
    """Raised when JSON parsing or schema validation fails after all retry attempts."""
    def __init__(self, message: str, raw_text: str = "", details: Optional[str] = None):
        super().__init__(message)
        self.raw_text = raw_text
        self.details = details or message


def strip_fences(text: str) -> str:
    """
    Strips markdown code fences (```json ... ``` or ``` ... ```)
    and returns the innermost candidate JSON substring.
    """
    cleaned = text.strip()

    # If wrapped in code blocks, extract content inside
    fence_pattern = r"^```(?:json)?\s*\n?(.*?)\n?```$"
    match = re.search(fence_pattern, cleaned, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned = match.group(1).strip()

    # Find outermost curly braces { ... }
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        return cleaned[first_brace : last_brace + 1].strip()

    return cleaned


def _sanitize_json_string(text: str) -> str:
    """
    Removes raw control characters (0x00-0x1F except \\t \\n \\r) that cause
    json.JSONDecodeError 'Invalid control character'. LLMs sometimes embed them
    in the body field of their JSON output.
    """
    # Replace bare (unescaped) control chars except tab/newline/CR with a space
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    return sanitized


def parse_raw_json(text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON dictionary from text.
    Applies fence stripping, control-character sanitization, and common repairs.
    """
    stripped = strip_fences(text)
    sanitized = _sanitize_json_string(stripped)

    for candidate in (sanitized, stripped):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    # Attempt further repair: single-quote keys/values, Python literals
    try:
        repaired = re.sub(r"(?<=[{,\s])'([^']+)'(?=\s*:)", r'"\1"', sanitized)
        repaired = re.sub(r":\s*'([^']*)'", r': "\1"', repaired)
        repaired = repaired.replace("None", "null").replace("True", "true").replace("False", "false")
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Last-resort: try to decode as JSON with strict=False (allows control chars)
    try:
        data = json.loads(sanitized, strict=False)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Raise informative error from the sanitized version
    try:
        json.loads(sanitized)
    except json.JSONDecodeError as err:
        raise ValueError(f"Invalid JSON: {err.msg} at line {err.lineno} col {err.colno}") from err
    raise ValueError(f"Expected JSON object (dict), got non-dict after parsing")


def extract_json(
    raw_text: str,
    schema_cls: Type[T],
    prompt_context: Optional[str] = None,
    system_context: Optional[str] = None,
    max_retries: int = 2,
    generate_fn: Optional[Callable[[str, Optional[str]], str]] = None,
) -> T:
    """
    Extracts and validates JSON from raw_text against schema_cls.
    If parsing/validation fails, retries up to max_retries using generate_fn.
    """
    current_text = raw_text
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            parsed_dict = parse_raw_json(current_text)
            validated = schema_cls.model_validate(parsed_dict)
            return validated
        except (ValueError, ValidationError, json.JSONDecodeError) as err:
            last_error = err
            logger.warning(
                f"JSON extraction attempt {attempt + 1}/{max_retries + 1} failed: {err}"
            )

            # If we have retries left and a generation function available, ask LLM to fix it
            if attempt < max_retries and generate_fn is not None:
                retry_prompt = (
                    f"CRITICAL ERROR: Your previous response was not valid JSON or failed schema validation.\n"
                    f"Error details: {err}\n"
                    f"Please output ONLY the corrected, valid JSON object enclosed in {{}}. No conversational text, no markdown explanation.\n"
                )
                if prompt_context:
                    retry_prompt += f"\nOriginal Request:\n{prompt_context}\n"
                retry_prompt += f"\nMalformed output to fix:\n{current_text}\n"

                current_text = generate_fn(retry_prompt, system_context)

    # Final failure: raise structured exception
    error_msg = f"Failed to extract valid {schema_cls.__name__} after {max_retries + 1} attempts: {last_error}"
    raise JSONExtractionError(error_msg, raw_text=current_text, details=str(last_error))


def safe_extract_json(
    raw_text: str,
    schema_cls: Type[T],
    prompt_context: Optional[str] = None,
    system_context: Optional[str] = None,
    max_retries: int = 2,
    generate_fn: Optional[Callable[[str, Optional[str]], str]] = None,
) -> Tuple[Optional[T], Optional[str]]:
    """
    Safe wrapper that returns (instance, None) on success
    or (None, error_message) on failure (never crashes GUI).
    """
    try:
        instance = extract_json(
            raw_text=raw_text,
            schema_cls=schema_cls,
            prompt_context=prompt_context,
            system_context=system_context,
            max_retries=max_retries,
            generate_fn=generate_fn,
        )
        return instance, None
    except JSONExtractionError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected error during JSON extraction: {exc}"
