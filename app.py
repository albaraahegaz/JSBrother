"""
JSBrother — Streamlit GUI (M8)
Calls existing orchestrator / writer / actions. Never changes score or verdict.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import streamlit as st

import config
from jsb.agents.action_agent import approve_and_send, freelancer_save_and_log, save_draft
from jsb.agents.profile_analyzer import analyze_profile, load_saved_profile
from jsb.agents.writer import refine_draft, write_draft
from jsb.gui_helpers import (
    DRAFT_LABELS,
    build_stepper_html,
    build_verdict_card_html,
    default_session_state,
    draft_options_for_mode,
    friendly_error,
    kinds_for_draft_choice,
    load_css_text,
    posting_too_short,
    render_chips,
    reset_analysis_keys,
    score_gauge_svg,
    tracker_metrics,
    verdict_color,
)
from jsb import llm_client as llm_client_mod
from jsb.llm_client import LLMClient
from jsb.memory import SessionMemory
from jsb.orchestrator import run as run_pipeline
from jsb.schemas import Draft, MatchResult, ProfileJSON
from jsb.tools.tracker import read_tracker

BASE = Path(__file__).resolve().parent


def _sync_llm_mode() -> LLMClient:
    """Point singleton + config at mock/remote per sidebar (GUI only; agents unchanged)."""
    mode = "mock" if st.session_state.mock_mode else "remote"
    config.LLM_MODE = mode
    if st.session_state.llm_base_url:
        config.LLM_BASE_URL = st.session_state.llm_base_url.rstrip("/")
    if st.session_state.llm_api_key:
        config.LLM_API_KEY = st.session_state.llm_api_key
    client = LLMClient(
        mode=mode,
        base_url=config.LLM_BASE_URL,
        api_key=config.LLM_API_KEY or st.session_state.llm_api_key,
    )
    llm_client_mod._default_client = client
    return client


def _init_state() -> None:
    defaults = default_session_state()
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Seed settings from env once (never display secrets in logs)
    if not st.session_state.llm_base_url:
        st.session_state.llm_base_url = config.LLM_BASE_URL or ""
    if not st.session_state.smtp_user:
        st.session_state.smtp_user = config.SMTP_USER or ""
    if st.session_state.mock_mode is True and config.LLM_MODE == "remote":
        # Prefer mock for first-load safety unless user already toggled
        pass


def _inject_css() -> None:
    css = load_css_text(BASE / "assets" / "style.css")
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _client() -> LLMClient:
    return _sync_llm_mode()


def _apply_profile_settings(profile: ProfileJSON) -> ProfileJSON:
    if st.session_state.mode == "project":
        profile.min_budget = float(st.session_state.min_budget) if st.session_state.min_budget else None
        profile.min_lead_days = int(st.session_state.min_lead_days or 7)
    return profile


def _ensure_profile() -> Optional[ProfileJSON]:
    if st.session_state.profile is not None:
        return _apply_profile_settings(st.session_state.profile)
    saved = load_saved_profile()
    if saved:
        st.session_state.profile = saved
        st.session_state.profile_source = "saved"
        return _apply_profile_settings(saved)
    return None


def _render_hero() -> None:
    mode = st.session_state.mode
    badge = (
        '<span class="jsb-badge jsb-badge-freelance">Freelancer</span>'
        if mode == "project"
        else '<span class="jsb-badge jsb-badge-job">Job Seeker</span>'
    )
    st.markdown(
        f"""
        <div class="jsb-hero">
          <h1 class="jsb-hero-title">JSBrother</h1>
          <p class="jsb-hero-tag">Your AI brother for applications &amp; proposals — score, draft, send.</p>
          {badge}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_verdict_card(match: MatchResult, mode: str) -> None:
    fl = match.freelancer if mode == "project" else None
    html = build_verdict_card_html(
        match.score,
        match.verdict,
        match.matched or [],
        match.missing_must or [],
        match.reasons or [],
        match.advice or "",
        budget_fit=fl.budget_fit if fl else None,
        deadline_fit=fl.deadline_fit if fl else None,
        red_flags=fl.red_flags if fl else None,
        deal_breaker_hits=match.deal_breaker_hits or [],
        warnings=match.warnings or [],
    )
    # strip indentation so Markdown doesn't treat the HTML as a code block
    html = "".join(line.strip() for line in html.splitlines() if line.strip())
    st.markdown(html, unsafe_allow_html=True)

def _render_trace(trace: Optional[dict]) -> None:
    if not trace:
        return
    with st.expander("How I decided", expanded=False):
        steps = trace.get("steps") or []
        for s in steps:
            name = s.get("step", "?")
            ms = s.get("ms", 0)
            extra = {k: v for k, v in s.items() if k not in ("step", "ms")}
            st.markdown(f"- **{name}** — `{ms} ms` " + (f"`{extra}`" if extra else ""))
        total = trace.get("total_ms")
        if total is not None:
            st.caption(f"Total pipeline: {total} ms · LLM mode: {trace.get('llm_mode')}")
        retrieved = (trace.get("retrieved_chunks") or {}).get("playbook_rules") or []
        if retrieved:
            st.markdown("**Playbook rules used**")
            for r in retrieved[:5]:
                st.markdown(f"- {r}")


def _run_analysis(force: bool = False) -> None:
    st.session_state.last_error = None
    posting_text = (st.session_state.posting_text or "").strip()
    posting_url = (st.session_state.posting_url or "").strip()
    source = posting_url if posting_url else posting_text

    if not posting_text and not posting_url:
        st.session_state.last_error = "Posting text is empty. Paste a job or project description first."
        return

    profile = _ensure_profile()
    if profile is None and not st.session_state.get("_pending_cv_bytes"):
        st.session_state.last_error = "No profile loaded. Upload a CV or reuse the saved profile."
        return

    client = _client()
    memory = SessionMemory()

    try:
        st.session_state.analyze_step = "reading"
        # Profile from upload pending in this run
        if st.session_state.profile is None and st.session_state.get("_cv_temp_path"):
            profile = analyze_profile(
                st.session_state["_cv_temp_path"],
                min_budget=st.session_state.min_budget if st.session_state.mode == "project" else None,
                min_lead_days=int(st.session_state.min_lead_days or 7),
                save_to_disk=True,
            )
            st.session_state.profile = profile
            st.session_state.profile_source = "upload"

        profile = _apply_profile_settings(_ensure_profile())
        if profile is None:
            st.session_state.last_error = "No profile loaded. Upload a CV or reuse the saved profile."
            return

        st.session_state.analyze_step = "matching"
        # Prefer pasted text; URL alone also works via posting_analyzer/web_fetcher
        posting_input: Any = source

        st.session_state.analyze_step = "writing"
        result = run_pipeline(
            profile=profile,
            posting=posting_input,
            mode=st.session_state.mode,
            force=force,
            memory=memory,
            client=client,
            skip_analyzers=False,
        )
        st.session_state.result = result
        st.session_state.match = result.match
        st.session_state.trace = result.trace
        st.session_state.skipped = result.skipped
        st.session_state.analyzed = True
        st.session_state.analyze_step = "done"
        if result.draft is not None:
            st.session_state.draft = result.draft
            st.session_state.draft_subject = result.draft.subject
            st.session_state.draft_body = result.draft.body
        else:
            st.session_state.draft = None
            st.session_state.draft_subject = ""
            st.session_state.draft_body = ""
    except Exception as exc:
        st.session_state.last_error = friendly_error(exc)
        st.session_state.analyze_step = None


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### Controls")
        mode_choice = st.radio(
            "Mode",
            options=["Job Seeker", "Freelancer"],
            index=0 if st.session_state.mode == "job" else 1,
        )
        st.session_state.mode = "project" if mode_choice == "Freelancer" else "job"

        st.session_state.mock_mode = st.toggle(
            "Mock mode (no remote LLM)",
            value=bool(st.session_state.mock_mode),
            help="Use canned responses so the app works without the GPU endpoint.",
        )
        st.session_state.dry_run = st.toggle(
            "Dry-run email (ON by default)",
            value=bool(st.session_state.dry_run),
            help="When ON, Approve & Send logs but does not contact SMTP.",
        )

        if st.session_state.mode == "project":
            st.markdown("#### Freelancer settings")
            st.session_state.min_budget = st.number_input(
                "Min budget (USD)",
                min_value=0.0,
                value=float(st.session_state.min_budget or 500),
                step=50.0,
            )
            st.session_state.min_lead_days = st.number_input(
                "Min lead days",
                min_value=1,
                value=int(st.session_state.min_lead_days or 7),
                step=1,
            )

        st.divider()
        if st.button("Reset session", use_container_width=True):
            keep = {
                "mock_mode": st.session_state.mock_mode,
                "dry_run": st.session_state.dry_run,
                "llm_base_url": st.session_state.llm_base_url,
                "llm_api_key": st.session_state.llm_api_key,
                "smtp_user": st.session_state.smtp_user,
                "smtp_app_password": st.session_state.smtp_app_password,
            }
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            for k, v in keep.items():
                st.session_state[k] = v
            st.rerun()

        st.caption(
            f"LLM: {'mock' if st.session_state.mock_mode else 'remote'} · "
            f"Dry-run: {'ON' if st.session_state.dry_run else 'OFF'}"
        )


def _tab_analyze() -> None:
    if not st.session_state.onboarding_dismissed and not st.session_state.analyzed:
        st.markdown(
            """
            <div class="jsb-onboarding">
              <strong>Get started in 3 steps</strong>
              <ol>
                <li>Upload your CV (or reuse the saved profile).</li>
                <li>Paste a job or project posting, then click <b>Analyze</b>.</li>
                <li>Review the verdict, edit the draft, then Approve &amp; Send (or Download).</li>
              </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Got it"):
            st.session_state.onboarding_dismissed = True
            st.rerun()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### 1. Profile")
        saved = load_saved_profile()
        reuse = False
        if saved:
            st.caption(f"Saved profile: **{saved.name or 'Candidate'}** ({saved.years_experience} yrs)")
            reuse = st.checkbox("Reuse saved profile", value=st.session_state.profile_source == "saved" or st.session_state.profile is not None)
            if reuse:
                st.session_state.profile = saved
                st.session_state.profile_source = "saved"

        uploaded = st.file_uploader("Upload CV / profile (PDF or TXT)", type=["pdf", "txt"])
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.getvalue())
            tmp.close()
            st.session_state["_cv_temp_path"] = tmp.name
            try:
                _sync_llm_mode()
                with st.spinner("Reading CV…"):
                    profile = analyze_profile(
                        tmp.name,
                        min_budget=st.session_state.min_budget if st.session_state.mode == "project" else None,
                        min_lead_days=int(st.session_state.min_lead_days or 7),
                        save_to_disk=True,
                    )
                st.session_state.profile = profile
                st.session_state.profile_source = "upload"
                st.success(f"Profile loaded: {profile.name or 'Candidate'}")
            except Exception as exc:
                st.error(friendly_error(exc))

        paste_profile = st.text_area(
            "Or paste profile text",
            height=100,
            placeholder="Optional paste fallback if PDF fails…",
        )
        if paste_profile.strip() and st.button("Parse pasted profile"):
            try:
                _sync_llm_mode()
                profile = analyze_profile(
                    paste_profile,
                    min_budget=st.session_state.min_budget if st.session_state.mode == "project" else None,
                    min_lead_days=int(st.session_state.min_lead_days or 7),
                    save_to_disk=True,
                )
                st.session_state.profile = profile
                st.session_state.profile_source = "paste"
                st.success(f"Profile loaded: {profile.name or 'Candidate'}")
            except Exception as exc:
                st.error(friendly_error(exc))

    with col_r:
        st.markdown("#### 2. Posting")
        st.session_state.posting_text = st.text_area(
            "Paste job / project posting",
            value=st.session_state.posting_text,
            height=220,
            placeholder="Paste the full posting text here…",
        )
        st.session_state.posting_url = st.text_input(
            "Optional URL",
            value=st.session_state.posting_url,
            placeholder="https://… (many sites block scrapers — paste is preferred)",
        )

    st.markdown(build_stepper_html(st.session_state.analyze_step), unsafe_allow_html=True)

    # Draft type selector (used on Analyze + regenerate)
    if st.session_state.mode == "project":
        draft_opts = ["Proposal", "Short bid message", "Both"]
        default_choice = "Proposal"
    else:
        draft_opts = ["Cover letter", "Email message", "Both"]
        default_choice = "Cover letter"
    if st.session_state.draft_type_choice not in draft_opts:
        st.session_state.draft_type_choice = default_choice
    st.session_state.draft_type_choice = st.selectbox(
        "Draft type",
        options=draft_opts,
        index=draft_opts.index(st.session_state.draft_type_choice),
        help="Both generates two drafts you can refine separately.",
    )

    analyze_clicked = st.button("Analyze", type="primary", use_container_width=True)
    if analyze_clicked:
        st.session_state.onboarding_dismissed = True
        pasted = (st.session_state.posting_text or "").strip()
        if pasted and posting_too_short(pasted) and not (st.session_state.posting_url or "").strip():
            st.info("Paste the full posting text")
        else:
            with st.status("Analyzing…", expanded=True) as status:
                st.write("1. Reading CV / profile…")
                st.session_state.analyze_step = "reading"
                st.write("2. Matching skills & computing verdict…")
                st.session_state.analyze_step = "matching"
                st.write("3. Writing draft (if not SKIP)…")
                st.session_state.analyze_step = "writing"
                _run_analysis(force=False)
                if st.session_state.last_error:
                    status.update(label="Analysis failed", state="error")
                else:
                    status.update(label="Done", state="complete")

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    if not st.session_state.analyzed:
        if not st.session_state.last_error:
            st.markdown(
                '<div class="jsb-empty"><strong>No analysis yet</strong><br/>'
                "Upload a profile, paste a posting, then hit Analyze.</div>",
                unsafe_allow_html=True,
            )
        return

    match: MatchResult = st.session_state.match
    _render_verdict_card(match, st.session_state.mode)
    _render_trace(st.session_state.trace)

    # SKIP gate: only card + Apply anyway
    if st.session_state.skipped and st.session_state.draft is None:
        st.warning("Verdict is SKIP — draft was not generated. You can still apply anyway.")
        if st.button("Apply anyway", type="primary"):
            with st.spinner("Generating draft anyway…"):
                _run_analysis(force=True)
            st.rerun()
        return

    # Editable draft
    if st.session_state.draft is not None or st.session_state.draft_body:
        st.markdown("#### Draft")
        st.session_state.draft_subject = st.text_input(
            "Subject",
            value=st.session_state.draft_subject,
        )
        st.session_state.draft_body = st.text_area(
            "Body",
            value=st.session_state.draft_body,
            height=320,
        )
        draft_obj = st.session_state.draft or Draft(
            kind="proposal" if st.session_state.mode == "project" else "cover_letter",
            subject=st.session_state.draft_subject,
            body=st.session_state.draft_body,
        )
        # Keep edits in sync
        draft_obj.subject = st.session_state.draft_subject
        draft_obj.body = st.session_state.draft_body
        st.session_state.draft = draft_obj

        if draft_obj.flagged_claims:
            st.warning("Flagged claims: " + ", ".join(draft_obj.flagged_claims))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("Shorter", use_container_width=True):
                _do_refine("shorter")
        with c2:
            if st.button("More formal", use_container_width=True):
                _do_refine("formal")
        with c3:
            if st.button("Friendlier", use_container_width=True):
                _do_refine("friendlier")
        with c4:
            custom = st.text_input("Custom refine", placeholder="e.g. emphasize RAG…", label_visibility="collapsed")
            if st.button("Apply refine", use_container_width=True) and custom.strip():
                _do_refine(custom.strip())

        st.markdown("#### Approve")
        recipient = st.text_input(
            "Recipient email",
            value=(st.session_state.profile.email if st.session_state.profile else "") or "",
            placeholder="hiring@company.com",
        )

        if st.session_state.mode == "project":
            b1, b2 = st.columns(2)
            with b1:
                if st.button("Download / save proposal", type="primary", use_container_width=True):
                    _do_freelancer_save(recipient)
            with b2:
                if st.button("Email proposal (optional)", use_container_width=True):
                    if not recipient.strip() and not st.session_state.dry_run:
                        st.error("Add a recipient, or turn Dry-run ON.")
                    else:
                        _do_approve(recipient)
        else:
            if st.button("Approve & Send", type="primary", use_container_width=True):
                if not recipient.strip() and not st.session_state.dry_run:
                    if not (st.session_state.smtp_user or config.SMTP_USER):
                        st.error("Missing SMTP credentials. Add them in Settings, or keep Dry-run ON.")
                    else:
                        st.error("Enter a recipient email, or enable Dry-run.")
                else:
                    _do_approve(recipient)

        if st.session_state.action_status:
            st.info(st.session_state.action_status)


def _do_refine(instruction: str) -> None:
    try:
        profile = _ensure_profile()
        result = st.session_state.result
        posting = result.posting if result else None
        if not profile or not posting or not st.session_state.draft:
            st.session_state.last_error = "Nothing to refine yet — run Analyze first."
            return
        with st.spinner("Refining draft…"):
            refined = refine_draft(
                st.session_state.draft,
                instruction,
                profile,
                posting,
                match_result=st.session_state.match,
                client=_client(),
            )
        st.session_state.draft = refined
        st.session_state.draft_subject = refined.subject
        st.session_state.draft_body = refined.body
        st.rerun()
    except Exception as exc:
        st.error(friendly_error(exc))


def _sync_draft() -> Draft:
    d = st.session_state.draft or Draft(
        kind="proposal" if st.session_state.mode == "project" else "cover_letter"
    )
    d.subject = st.session_state.draft_subject
    d.body = st.session_state.draft_body
    return d


def _do_approve(recipient: str) -> None:
    try:
        profile = _ensure_profile()
        result = st.session_state.result
        if not profile or not result or not result.match:
            st.error("Run Analyze first.")
            return
        draft = _sync_draft()
        out = approve_and_send(
            profile=profile,
            posting=result.posting,
            match=result.match,
            draft=draft,
            recipient=recipient.strip() or None,
            dry_run=bool(st.session_state.dry_run),
            mode=st.session_state.mode,
        )
        st.session_state.action_status = (
            f"Logged · email_status={out['email_status']} · draft={out['draft_path']}"
        )
    except Exception as exc:
        st.error(friendly_error(exc))


def _do_freelancer_save(recipient: str) -> None:
    try:
        profile = _ensure_profile()
        result = st.session_state.result
        if not profile or not result:
            st.error("Run Analyze first.")
            return
        draft = _sync_draft()
        out = freelancer_save_and_log(
            profile=profile,
            posting=result.posting,
            match=result.match,
            draft=draft,
            email_to=recipient.strip() or None,
            dry_run=True if not recipient.strip() else bool(st.session_state.dry_run),
        )
        path = out["draft_path"]
        st.session_state.action_status = f"Saved · {path}"
        # Offer download
        data = Path(path).read_text(encoding="utf-8")
        st.download_button(
            "Download file",
            data=data,
            file_name=Path(path).name,
            mime="text/plain",
        )
    except Exception as exc:
        st.error(friendly_error(exc))


def _tab_tracker() -> None:
    rows = read_tracker()
    metrics = tracker_metrics(rows)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(
            f'<div class="jsb-metric"><div class="n">{metrics["applications"]}</div>'
            f'<div class="l">Applications</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f'<div class="jsb-metric"><div class="n">{metrics["avg_score"]}</div>'
            f'<div class="l">Avg score</div></div>',
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f'<div class="jsb-metric"><div class="n">{metrics["sent_count"]}</div>'
            f'<div class="l">Sent / dry-run</div></div>',
            unsafe_allow_html=True,
        )

    dist = metrics["verdict_dist"]
    st.markdown("##### Verdict distribution")
    if sum(dist.values()) == 0:
        st.markdown(
            '<div class="jsb-empty"><strong>Tracker is empty</strong><br/>'
            "Approve a draft to log your first application.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.bar_chart({"APPLY": dist["APPLY"], "CAUTION": dist["CAUTION"], "SKIP": dist["SKIP"]})

    f1, f2 = st.columns(2)
    with f1:
        mode_f = st.selectbox("Filter mode", ["All", "job", "project"])
    with f2:
        verd_f = st.selectbox("Filter verdict", ["All", "APPLY", "CAUTION", "SKIP"])

    filtered = rows
    if mode_f != "All":
        filtered = [r for r in filtered if (r.get("mode") or "") == mode_f]
    if verd_f != "All":
        filtered = [r for r in filtered if (r.get("verdict") or "").upper() == verd_f]

    if filtered:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        # CSV download
        import csv

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=list(filtered[0].keys()))
        writer.writeheader()
        writer.writerows(filtered)
        st.download_button(
            "Download filtered CSV",
            data=buf.getvalue(),
            file_name="jsbrother_tracker.csv",
            mime="text/csv",
        )
    elif rows:
        st.info("No rows match these filters.")


def _tab_settings() -> None:
    st.markdown("#### LLM endpoint")
    st.session_state.llm_base_url = st.text_input(
        "LLM base URL (ngrok)",
        value=st.session_state.llm_base_url,
        type="default",
    )
    st.session_state.llm_api_key = st.text_input(
        "LLM API key",
        value=st.session_state.llm_api_key or config.LLM_API_KEY,
        type="password",
    )
    if st.button("Test LLM connection"):
        try:
            client = _client()
            if client.mode == "mock":
                ping = client.generate("Respond with ONLY the word PONG.", max_new_tokens=8)
                st.success(f"Mock mode OK · response: {ping.strip()[:40]}")
            else:
                health = client.check_health()
                st.success(f"Remote healthy · {health}")
        except Exception as exc:
            st.error(friendly_error(exc))

    st.markdown("#### Email (Gmail SMTP)")
    st.session_state.smtp_user = st.text_input(
        "SMTP email",
        value=st.session_state.smtp_user or config.SMTP_USER,
    )
    st.session_state.smtp_app_password = st.text_input(
        "SMTP app password",
        value=st.session_state.smtp_app_password or "",
        type="password",
        help="Leave blank to keep the value from .env (never shown).",
    )
    if st.button("Test SMTP settings"):
        user = st.session_state.smtp_user or config.SMTP_USER
        pwd = st.session_state.smtp_app_password or config.SMTP_APP_PASSWORD
        if not user or not pwd:
            st.error("Missing SMTP email or app password. Add them here or in .env.")
        else:
            # Apply to config for this process only (do not print)
            config.SMTP_USER = user
            if st.session_state.smtp_app_password:
                config.SMTP_APP_PASSWORD = st.session_state.smtp_app_password
            from jsb.tools.email_sender import send_email

            res = send_email(
                to_addr=user,
                subject="[JSBrother] SMTP test",
                body="Dry-run connectivity check from Settings.",
                dry_run=True,
            )
            if res.status == "dry_run":
                st.success("SMTP credentials present · dry-run send OK (no email delivered).")
            else:
                st.error(res.detail or res.status)

    st.caption("Secrets stay in this session / .env. They are never written to the tracker.")


def main() -> None:
    st.set_page_config(
        page_title="JSBrother",
        page_icon="✨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    _inject_css()
    # Sync SMTP overrides into config for action agent
    if st.session_state.smtp_user:
        config.SMTP_USER = st.session_state.smtp_user
    if st.session_state.smtp_app_password:
        config.SMTP_APP_PASSWORD = st.session_state.smtp_app_password
    config.EMAIL_DRY_RUN = bool(st.session_state.dry_run)

    _sidebar()
    _render_hero()

    tab_a, tab_t, tab_s = st.tabs(["Analyze", "Tracker", "Settings"])
    with tab_a:
        _tab_analyze()
    with tab_t:
        _tab_tracker()
    with tab_s:
        _tab_settings()


if __name__ == "__main__":
    main()
