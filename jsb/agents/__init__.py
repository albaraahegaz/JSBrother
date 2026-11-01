"""JSBrother agents package."""

from jsb.agents.profile_analyzer import analyze_profile, load_saved_profile
from jsb.agents.posting_analyzer import analyze_posting
from jsb.agents.matcher import match, compute_match_score_and_verdict
from jsb.agents.writer import write_draft, refine_draft
from jsb.agents.router import detect_mode, validate_inputs
from jsb.agents.action_agent import save_draft, approve_and_send, freelancer_save_and_log

__all__ = [
    "analyze_profile",
    "load_saved_profile",
    "analyze_posting",
    "match",
    "compute_match_score_and_verdict",
    "write_draft",
    "refine_draft",
    "detect_mode",
    "validate_inputs",
    "save_draft",
    "approve_and_send",
    "freelancer_save_and_log",
]
