"""
JSBrother — Matcher Module Re-export
"""

from jsb.agents.matcher import (
    match,
    compute_match_score_and_verdict,
    evaluate_deal_breakers,
    compute_freelancer_fit,
    build_default_explanations,
    generate_llm_explanations,
)

__all__ = [
    "match",
    "compute_match_score_and_verdict",
    "evaluate_deal_breakers",
    "compute_freelancer_fit",
    "build_default_explanations",
    "generate_llm_explanations",
]
