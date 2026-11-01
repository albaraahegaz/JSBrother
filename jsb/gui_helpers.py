"""
JSBrother — GUI helpers (M8)
Pure functions for SVG gauge, chips, friendly errors, and session defaults.
No Streamlit dependency so unit tests stay lightweight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERDICT_COLORS = {
    "APPLY": "#2E9E5B",
    "CAUTION": "#F2A33A",
    "SKIP": "#D64545",
}

CIRCUMFERENCE = 2 * 3.141592653589793 * 45  # r=45


def verdict_color(verdict: str) -> str:
    return VERDICT_COLORS.get((verdict or "").upper(), "#9AA3C7")


def gauge_offset(score: float) -> float:
    """stroke-dashoffset for a ring filled proportionally to score (0-100)."""
    s = max(0.0, min(100.0, float(score)))
    return CIRCUMFERENCE * (1.0 - s / 100.0)


def _one_line_html(html: str) -> str:
    """Collapse whitespace so Streamlit markdown never treats HTML as an indented code block."""
    return " ".join((html or "").split())


def score_gauge_svg(score: float, verdict: str, size: int = 140) -> str:
    """Animated circular SVG score gauge colored by verdict."""
    color = verdict_color(verdict)
    offset = gauge_offset(score)
    label = f"{int(round(score))}"
    return _one_line_html(
        f'<svg class="jsb-gauge" width="{size}" height="{size}" viewBox="0 0 120 120" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Match score {label}">'
        f'<circle cx="60" cy="60" r="45" fill="none" stroke="#2A3150" stroke-width="10"/>'
        f'<circle class="jsb-gauge-ring" cx="60" cy="60" r="45" fill="none" '
        f'stroke="{color}" stroke-width="10" stroke-linecap="round" '
        f'style="--gauge-offset: {offset:.2f}; stroke-dasharray: {CIRCUMFERENCE:.2f}; '
        f'stroke-dashoffset: {offset:.2f};"/>'
        f'<text x="60" y="56" text-anchor="middle" fill="#E8EAF6" '
        f'font-family="Inter,sans-serif" font-size="28" font-weight="700">{label}</text>'
        f'<text x="60" y="76" text-anchor="middle" fill="#9AA3C7" '
        f'font-family="Inter,sans-serif" font-size="11">/ 100</text>'
        f"</svg>"
    )


def render_chips(items: List[str], kind: str = "match") -> str:
    """HTML chip row. kind: match | missing | alert | danger."""
    cls = {
        "match": "jsb-chip-match",
        "missing": "jsb-chip-missing",
        "alert": "jsb-chip-alert",
        "danger": "jsb-chip-danger",
    }.get(kind, "jsb-chip-match")
    if not items:
        return '<div class="jsb-chip-row"><span class="jsb-chip jsb-chip-alert">None</span></div>'
    chips = "".join(
        f'<span class="jsb-chip {cls}">{_escape_html(str(x))}</span>' for x in items if str(x).strip()
    )
    return f'<div class="jsb-chip-row">{chips}</div>'


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def friendly_error(exc: BaseException) -> str:
    """Map exceptions to short user-facing messages (no tracebacks/secrets)."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()
    if "timeout" in msg or "timed out" in msg:
        return "The AI endpoint timed out. Check Settings → LLM URL, or switch to Mock mode."
    if "connection" in msg or "refused" in msg or "failed to communicate" in msg:
        return "Cannot reach the AI endpoint. Paste a fresh ngrok URL in Settings, or enable Mock mode."
    if "401" in msg or "403" in msg or "unauthorized" in msg or "api key" in msg:
        return "LLM authentication failed. Check the API key in Settings."
    if "empty" in msg and ("pdf" in msg or "cv" in msg or "profile" in msg):
        return "The CV/PDF looks empty or unreadable. Try another file or paste your profile text."
    if "pdf" in msg or "pypdf" in msg or "cannot analyze profile" in msg:
        return "Could not read that PDF. Upload a text-based CV or paste the content."
    if "posting" in msg and "empty" in msg:
        return "Posting text is empty. Paste a job or project description first."
    if "smtp" in msg or "credentials" in msg:
        return "Email credentials are missing. Add SMTP email + app password in Settings, or keep Dry-run ON."
    if "profile is missing" in msg or "no profile" in msg:
        return "No profile loaded. Upload a CV or reuse the saved profile."
    if name == "valueerror":
        return str(exc)[:240]
    return "Something went wrong while analyzing. Try Mock mode or check Settings."


def default_session_state() -> Dict[str, Any]:
    """Keys/defaults for st.session_state initialization."""
    return {
        "mode": "job",  # job | project
        "mock_mode": True,
        "dry_run": True,
        "min_budget": 500.0,
        "min_lead_days": 7,
        "profile": None,
        "profile_source": None,
        "posting_text": "",
        "posting_url": "",
        "result": None,  # PipelineResult-like dict or object
        "match": None,
        "draft": None,
        "draft_subject": "",
        "draft_body": "",
        "trace": None,
        "skipped": False,
        "analyzed": False,
        "onboarding_dismissed": False,
        "last_error": None,
        "action_status": None,
        "llm_base_url": "",
        "llm_api_key": "",
        "smtp_user": "",
        "smtp_app_password": "",
        "analyze_step": None,
        "draft_type_choice": "Cover letter",
        "drafts": {},  # kind -> Draft
        "active_draft_kind": None,
    }


def reset_analysis_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Clear analysis outputs while keeping settings/mode."""
    for k in (
        "result", "match", "draft", "draft_subject", "draft_body",
        "trace", "skipped", "analyzed", "last_error", "action_status", "analyze_step",
        "drafts", "active_draft_kind",
    ):
        if k in ("skipped", "analyzed"):
            state[k] = False
        elif k == "drafts":
            state[k] = {}
        elif k in ("draft_subject", "draft_body"):
            state[k] = ""
        else:
            state[k] = None
    return state


def tracker_metrics(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    """Compute applications, average score, sent count, verdict distribution."""
    n = len(rows)
    scores = []
    sent = 0
    dist = {"APPLY": 0, "CAUTION": 0, "SKIP": 0}
    for r in rows:
        try:
            scores.append(float(r.get("score") or 0))
        except ValueError:
            pass
        status = (r.get("email_status") or "").lower()
        if status in ("sent", "dry_run"):
            sent += 1
        v = (r.get("verdict") or "").upper()
        if v in dist:
            dist[v] += 1
    avg = sum(scores) / len(scores) if scores else 0.0
    return {
        "applications": n,
        "avg_score": round(avg, 1),
        "sent_count": sent,
        "verdict_dist": dist,
    }


def load_css_text(css_path: Optional[Path] = None) -> str:
    path = css_path or (Path(__file__).resolve().parent.parent / "assets" / "style.css")
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def mode_label(mode: str) -> str:
    return "Freelancer" if mode == "project" else "Job Seeker"


def build_stepper_html(active: Optional[str] = None) -> str:
    steps = ["Reading CV", "Matching", "Writing"]
    key_map = {"reading": 0, "matching": 1, "writing": 2, "done": 3}
    idx = key_map.get((active or "").lower(), -1)
    parts = []
    for i, label in enumerate(steps):
        cls = "jsb-step"
        if idx > i or active == "done":
            cls += " jsb-step-done"
        elif idx == i:
            cls += " jsb-step-active"
        parts.append(f'<span class="{cls}">{i + 1}. {label}</span>')
    return f'<div class="jsb-stepper">{"".join(parts)}</div>'


def build_verdict_card_html(
    score: float,
    verdict: str,
    matched: List[str],
    missing_must: List[str],
    reasons: List[str],
    advice: str,
    *,
    budget_fit: Optional[float] = None,
    deadline_fit: Optional[float] = None,
    red_flags: Optional[List[str]] = None,
    deal_breaker_hits: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> str:
    """
    Build verdict card HTML with NO leading indentation so Streamlit markdown
    does not treat the block as a fenced code sample.
    """
    gauge = score_gauge_svg(score, verdict)
    reasons_html = "".join(
        f'<div class="jsb-reason"><span class="jsb-reason-icon">&#9656;</span>'
        f"<span>{_escape_html(str(r))}</span></div>"
        for r in (reasons or [])[:3]
    )
    fl_html = ""
    if budget_fit is not None or deadline_fit is not None:
        bf = f"{budget_fit:.0%}" if budget_fit is not None else "—"
        df = f"{deadline_fit:.0%}" if deadline_fit is not None else "—"
        fl_html = (
            '<div style="margin-top:0.6rem;display:flex;gap:0.8rem;flex-wrap:wrap;'
            'font-size:0.85rem;color:#9AA3C7;">'
            f'<span>Budget fit: <b style="color:#E8EAF6">{bf}</b></span>'
            f'<span>Deadline fit: <b style="color:#E8EAF6">{df}</b></span></div>'
        )
        if red_flags:
            fl_html += (
                '<div style="margin-top:0.35rem"><small style="color:#9AA3C7">Red flags</small>'
                + render_chips(red_flags, "alert")
                + "</div>"
            )
    alerts = ""
    if deal_breaker_hits:
        alerts += (
            '<div><small style="color:#9AA3C7">Deal-breakers</small>'
            + render_chips(deal_breaker_hits, "danger")
            + "</div>"
        )
    if warnings:
        alerts += (
            '<div><small style="color:#9AA3C7">Warnings</small>'
            + render_chips(warnings, "alert")
            + "</div>"
        )
    advice_safe = _escape_html(advice or "")
    return _one_line_html(
        '<div class="jsb-card">'
        '<div style="display:flex;gap:1.4rem;align-items:center;flex-wrap:wrap;">'
        f"<div>{gauge}</div>"
        '<div style="flex:1;min-width:200px;">'
        f'<span class="jsb-verdict jsb-verdict-{verdict}">{verdict}</span>'
        '<div style="margin-top:0.85rem;">'
        '<small style="color:#9AA3C7">Matched skills</small>'
        f"{render_chips(matched[:12], 'match')}"
        '<small style="color:#9AA3C7">Missing must-haves</small>'
        f"{render_chips(missing_must[:12], 'missing')}"
        "</div></div></div>"
        f"{fl_html}{alerts}"
        f'<div style="margin-top:0.7rem;">{reasons_html}</div>'
        f'<div class="jsb-advice"><b>Advice:</b> {advice_safe}</div>'
        "</div>"
    )


DRAFT_LABELS = {
    "cover_letter": "Cover letter",
    "email_message": "Email message",
    "proposal": "Proposal",
    "short_bid": "Short bid message",
}


def draft_options_for_mode(mode: str) -> List[str]:
    if mode == "project":
        return ["Proposal", "Short bid message", "Both"]
    return ["Cover letter", "Email message", "Both"]


def kinds_for_draft_choice(mode: str, choice: str) -> List[str]:
    c = (choice or "").strip()
    if mode == "project":
        if c == "Short bid message":
            return ["short_bid"]
        if c == "Both":
            return ["proposal", "short_bid"]
        return ["proposal"]
    if c == "Email message":
        return ["email_message"]
    if c == "Both":
        return ["cover_letter", "email_message"]
    return ["cover_letter"]


def posting_too_short(text: str, min_words: int = 20) -> bool:
    return len((text or "").split()) < min_words
