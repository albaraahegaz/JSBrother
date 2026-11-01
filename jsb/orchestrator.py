"""
JSBrother — LCEL Orchestrator (M6)
Pipeline: Router → parallel Profile/Posting analyzers → Matcher → Gate → Writer → Claim Checker → Remember
Python alone decides score/verdict; LLM extracts, phrases, and writes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union

from langchain_core.runnables import RunnableLambda, RunnableParallel

import config
from jsb.agents.matcher import match as match_profile_posting
from jsb.agents.posting_analyzer import analyze_posting, get_extraction_path
from jsb.agents.profile_analyzer import analyze_profile, load_saved_profile
from jsb.agents.router import detect_mode, validate_inputs
from jsb.agents.writer import write_draft
from jsb.llm_client import LLMClient, get_client
from jsb.memory import SessionMemory
from jsb.schemas import Draft, MatchResult, PostingJSON, ProfileJSON
from jsb.tools.claim_checker import check_claims

logger = logging.getLogger("jsbrother.orchestrator")


@dataclass
class PipelineResult:
    mode: str
    profile: ProfileJSON
    posting: PostingJSON
    match: MatchResult
    draft: Optional[Draft]
    skipped: bool
    force: bool
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "verdict": self.match.verdict,
            "score": self.match.score,
            "skipped": self.skipped,
            "force": self.force,
            "draft": self.draft.model_dump() if self.draft and hasattr(self.draft, "model_dump") else (
                self.draft.dict() if self.draft else None
            ),
            "match": self.match.model_dump() if hasattr(self.match, "model_dump") else self.match.dict(),
            "trace": self.trace,
        }


def _timed(step_name: str, trace_steps: List[Dict[str, Any]], fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    trace_steps.append({"step": step_name, "ms": elapsed_ms})
    return result


def _ensure_profile(
    profile: Optional[Union[ProfileJSON, str]],
    memory: SessionMemory,
    reanalyze: bool = False,
) -> ProfileJSON:
    if isinstance(profile, ProfileJSON):
        memory.profile = profile
        return profile
    if isinstance(profile, str) and profile.strip():
        # Path or raw CV text
        return analyze_profile(profile, save_to_disk=True)
    saved = memory.profile or load_saved_profile()
    if saved is None:
        raise ValueError("No profile provided and none saved at data/profile.json")
    return saved


def _ensure_posting(
    posting: Optional[Union[PostingJSON, str]],
    mode: str,
) -> PostingJSON:
    if isinstance(posting, PostingJSON):
        if posting.type not in ("job", "project"):
            posting.type = "project" if mode == "project" else "job"
        elif mode in ("job", "project") and posting.type != mode:
            # Explicit mode wins when caller forces it
            posting.type = mode
        return posting
    if isinstance(posting, str) and posting.strip():
        p = analyze_posting(posting)
        p.type = mode if mode in ("job", "project") else p.type
        return p
    raise ValueError("Posting is required (PostingJSON or text/path/URL)")


def run(
    profile: Optional[Union[ProfileJSON, str]] = None,
    posting: Optional[Union[PostingJSON, str]] = None,
    mode: Optional[str] = None,
    force: bool = False,
    memory: Optional[SessionMemory] = None,
    client: Optional[LLMClient] = None,
    skip_analyzers: bool = False,
    draft_kind: Optional[str] = None,
) -> PipelineResult:
    """
    End-to-end pipeline. Returns verdict (+ draft unless SKIP and not force).

    Args:
        profile: ProfileJSON, CV path/text, or None to reuse saved profile.
        posting: PostingJSON or posting text/path/URL.
        mode: 'job' | 'project' | None (auto-detect).
        force: If True, generate draft even when verdict is SKIP ("Apply anyway").
        memory: Optional SessionMemory; created if omitted.
        client: Optional LLM client.
        skip_analyzers: If True and both are already structured objects, skip LLM extract.
        draft_kind: Optional writer kind (cover_letter/email_message/proposal/short_bid).
    """
    memory = memory or SessionMemory()
    client = client or get_client()
    steps: List[Dict[str, Any]] = []
    retrieved: Dict[str, Any] = {"playbook_rules": [], "cv_chunks": []}
    t_pipeline = time.perf_counter()

    # --- 1. Router ---
    validation = validate_inputs(profile, posting)
    if not validation["ok"] and not (
        isinstance(profile, ProfileJSON) or memory.profile or load_saved_profile()
    ):
        raise ValueError("; ".join(validation["errors"]))

    posting_text_hint = ""
    if isinstance(posting, str):
        posting_text_hint = posting
    elif isinstance(posting, PostingJSON):
        posting_text_hint = f"{posting.title} {posting.summary}"

    detected = _timed(
        "router",
        steps,
        detect_mode,
        posting_text=posting_text_hint,
        posting=posting if isinstance(posting, PostingJSON) else None,
        mode=mode,
        client=client,
    )

    # --- 2. Parallel analyzers (or reuse structured inputs) ---
    def _profile_branch(_: Dict[str, Any]) -> ProfileJSON:
        return _ensure_profile(profile, memory)

    def _posting_branch(_: Dict[str, Any]) -> PostingJSON:
        return _ensure_posting(posting, detected)

    if skip_analyzers and isinstance(profile, ProfileJSON) and isinstance(posting, PostingJSON):
        profile_obj = profile
        posting_obj = posting
        posting_obj.type = detected
        steps.append({"step": "analyzers", "ms": 0.0, "note": "skipped (structured inputs)"})
    else:
        t0 = time.perf_counter()
        parallel = RunnableParallel(
            profile=RunnableLambda(_profile_branch),
            posting=RunnableLambda(_posting_branch),
        )
        analyzed = parallel.invoke({})
        profile_obj = analyzed["profile"]
        posting_obj = analyzed["posting"]
        posting_obj.type = detected
        steps.append({
            "step": "analyzers_parallel",
            "ms": round((time.perf_counter() - t0) * 1000, 1),
            "must_have_path": get_extraction_path(posting_obj),
        })

    memory.profile = profile_obj
    try:
        memory.save_profile(profile_obj)
    except Exception as e:
        logger.warning(f"Could not persist profile: {e}")

    extraction_path = get_extraction_path(posting_obj)

    # --- 3. Matcher (Python score/verdict) ---
    match_result = _timed(
        "matcher",
        steps,
        match_profile_posting,
        profile_obj,
        posting_obj,
        mode=detected,
        client=client,
    )

    # --- 4. Gate ---
    skipped = match_result.verdict == "SKIP" and not force
    draft: Optional[Draft] = None
    writer_trace: Dict[str, Any] = {}

    if skipped:
        steps.append({"step": "gate", "ms": 0.0, "action": "STOP", "verdict": "SKIP"})
    else:
        steps.append(
            {
                "step": "gate",
                "ms": 0.0,
                "action": "CONTINUE",
                "verdict": match_result.verdict,
                "force": force,
            }
        )
        # --- 5–6. Writer (+ claim checker inside write_draft) ---
        t0 = time.perf_counter()
        draft, writer_trace = write_draft(
            profile_obj,
            posting_obj,
            match_result,
            client=client,
            return_trace=True,
            draft_kind=draft_kind,
        )
        # Explicit claim re-check for the decision trace
        claims_ok, flagged = check_claims(draft.body, profile_obj, posting_obj, match_result)
        draft.claims_ok = claims_ok
        draft.flagged_claims = flagged
        steps.append(
            {
                "step": "writer_claim_checker",
                "ms": round((time.perf_counter() - t0) * 1000, 1),
                "claims_ok": claims_ok,
                "flagged": flagged,
                "regenerated": writer_trace.get("regenerated", False),
            }
        )
        retrieved["playbook_rules"] = writer_trace.get("playbook_rules", [])
        retrieved["cv_chunks"] = writer_trace.get("cv_chunks", [])

    # --- 7. Remember ---
    trace = {
        "steps": steps,
        "timings_ms": {s["step"]: s["ms"] for s in steps},
        "total_ms": round((time.perf_counter() - t_pipeline) * 1000, 1),
        "retrieved_chunks": retrieved,
        "mode": detected,
        "llm_mode": client.mode,
        "force": force,
        "skipped": skipped,
        "must_have_path": extraction_path,
    }
    memory.remember_run(
        posting=posting_obj,
        match=match_result,
        draft=draft,
        trace=trace,
        user_note=f"run mode={detected} title={posting_obj.title}",
    )
    steps.append({"step": "remember", "ms": 0.0})

    return PipelineResult(
        mode=detected,
        profile=profile_obj,
        posting=posting_obj,
        match=match_result,
        draft=draft,
        skipped=skipped,
        force=force,
        trace=trace,
    )


# Convenience alias matching the milestone wording
def run_pipeline(*args, **kwargs) -> PipelineResult:
    return run(*args, **kwargs)
