"""Unit tests for M8 GUI helpers (no Streamlit runtime)."""

from jsb.gui_helpers import (
    build_stepper_html,
    default_session_state,
    friendly_error,
    gauge_offset,
    render_chips,
    reset_analysis_keys,
    score_gauge_svg,
    tracker_metrics,
    verdict_color,
)


def test_verdict_colors():
    assert verdict_color("APPLY") == "#2E9E5B"
    assert verdict_color("CAUTION") == "#F2A33A"
    assert verdict_color("SKIP") == "#D64545"


def test_gauge_offset_bounds():
    assert gauge_offset(0) > gauge_offset(50) > gauge_offset(100)
    assert abs(gauge_offset(100)) < 1e-6 or gauge_offset(100) == 0.0
    svg = score_gauge_svg(80, "APPLY")
    assert "80" in svg
    assert "#2E9E5B" in svg
    assert "jsb-gauge-ring" in svg


def test_render_chips():
    html = render_chips(["python", "pytorch"], "match")
    assert "jsb-chip-match" in html
    assert "python" in html
    missing = render_chips(["cobol"], "missing")
    assert "jsb-chip-missing" in missing
    empty = render_chips([], "match")
    assert "None" in empty


def test_friendly_errors():
    assert "endpoint" in friendly_error(RuntimeError("Failed to communicate with remote")).lower()
    assert "empty" in friendly_error(ValueError("Posting text is empty")).lower() or "paste" in friendly_error(
        ValueError("Cannot analyze posting: Posting text is empty.")
    ).lower()
    assert "PDF" in friendly_error(ValueError("bad pdf parse")) or "pdf" in friendly_error(
        ValueError("Cannot analyze profile: CV text is empty.")
    ).lower()
    assert "SMTP" in friendly_error(RuntimeError("smtp credentials missing")) or "credentials" in friendly_error(
        RuntimeError("smtp credentials missing")
    ).lower()


def test_session_defaults_and_reset():
    state = default_session_state()
    assert state["dry_run"] is True
    assert state["mock_mode"] is True
    state["analyzed"] = True
    state["draft_body"] = "hello"
    state["match"] = object()
    reset_analysis_keys(state)
    assert state["analyzed"] is False
    assert state["draft_body"] == ""
    assert state["match"] is None
    assert state["dry_run"] is True  # settings preserved by caller; reset only analysis keys


def test_tracker_metrics():
    rows = [
        {"score": "80", "verdict": "APPLY", "email_status": "dry_run", "mode": "job"},
        {"score": "60", "verdict": "CAUTION", "email_status": "sent", "mode": "project"},
        {"score": "40", "verdict": "SKIP", "email_status": "", "mode": "job"},
    ]
    m = tracker_metrics(rows)
    assert m["applications"] == 3
    assert m["avg_score"] == 60.0
    assert m["sent_count"] == 2
    assert m["verdict_dist"]["APPLY"] == 1


def test_stepper_html():
    html = build_stepper_html("matching")
    assert "jsb-step-active" in html
    assert "Matching" in html
