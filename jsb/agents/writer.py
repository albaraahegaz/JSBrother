"""
JSBrother — Writer Agent & Refine Module (M5)
Generates tailored, high-converting application drafts per §7:
- Cover Letter (Job Seeker mode, ~220–300 words): Hook -> 2-3 achievements -> gap handling -> CTA
- Proposal (Freelancer mode, ~150–250 words): Problem in client's words -> 3-step approach -> timeline -> price -> proof point -> clarifying question
Uses:
- Top-3 Playbook rules retrieved from FAISS
- Top-3 CV chunks retrieved from FAISS
- Claim checking with automatic single-turn regeneration for ungrounded claims
"""

import json
import logging
from typing import Optional, List, Dict, Any, Tuple

from pydantic import BaseModel, Field
import config
from jsb.schemas import ProfileJSON, PostingJSON, MatchResult, Draft
from jsb.tools.vector_store import (
    build_playbook_retriever,
    build_cv_retriever,
    PlaybookRetriever,
    CVRetriever,
)
from jsb.tools.claim_checker import check_claims
from jsb.json_utils import extract_json
from jsb.llm_client import get_client, LLMClient

logger = logging.getLogger("jsbrother.writer")


class DraftContent(BaseModel):
    """Schema for LLM draft generation output."""
    kind: str = "cover_letter"
    subject: str = ""
    body: str = ""


EMAIL_MESSAGE_SYSTEM_PROMPT = """You write short, professional application emails.
Return ONLY valid JSON: {"kind":"email_message","subject":"...","body":"..."}.
Body: 80–140 words. Greeting → 2 sentences of fit → 1 proof point from CV → CTA + signature.
No invented facts. No markdown fences.
"""

SHORT_BID_SYSTEM_PROMPT = """You write short freelance bid messages (Upwork-style).
Return ONLY valid JSON: {"kind":"short_bid","subject":"...","body":"..."}.
Body: 60–120 words. Open with client's need → 2 concrete delivery points → price/timeline hint → one question?
No invented facts. No markdown fences.
"""


COVER_LETTER_SYSTEM_PROMPT = """You are an elite executive career coach and technical copywriter.
Your task is to write a compelling, tailored cover letter for a job applicant.

Strict Output Rules:
1. Return ONLY a valid JSON object:
{
  "kind": "cover_letter",
  "subject": "Clear email subject line (e.g. Application: Role Title - Candidate Name)",
  "body": "Full body of the cover letter with standard paragraphs"
}
2. Structure (exactly 4 sections, separate paragraphs with blank lines):
   - Section 1 (Hook): Immediately connect with the company's mission, product challenge, or team goals. No generic openings. Use words like "excited to apply" or "opportunity".
   - Section 2 (Core Value): Present 2 to 3 quantified achievements from the candidate's CV that prove capability in the posting's must-have requirements. Write 4-6 full sentences.
   - Section 3 (Gap Handling): If there are missing must-have or nice-to-have skills, handle them honestly and proactively (eager to deepen / learn). Never falsify skills. Write 2-3 sentences.
   - Section 4 (Call to Action): Confident, collaborative close requesting a technical conversation. End with "Sincerely," and the candidate name.
3. LENGTH IS MANDATORY: body MUST contain 240-280 words (hard floor 220, hard ceiling 300). Aim ~260. Under 220 = REJECTED. Expand achievements with concrete tools and outcomes from CV chunks.
4. Truthfulness: Never invent companies, years of experience, tools, or metrics not found in the verified CV chunks.
5. Do NOT include markdown code fences or text outside the JSON object.
"""

PROPOSAL_SYSTEM_PROMPT = """You are a top-rated independent consultant and technical proposal specialist.
Your task is to write a winning freelance proposal for a client project.

Strict Output Rules:
1. Return ONLY a valid JSON object with fields: "kind", "subject", "body".
   The body field must contain the full proposal as plain text (no nested JSON, no markdown).
2. The body MUST have ALL 6 sections in this exact order, each as its own paragraph:
   Section 1 – Problem Statement: Start with "I understand you need..." then restate the client's goal/challenge in 2-3 sentences using their words.
   Section 2 – 3-Step Technical Approach (use these exact labels on separate lines):
     Step 1: [first deliverable — 2 sentences]
     Step 2: [second deliverable — 2 sentences]
     Step 3: [third deliverable / validation — 2 sentences]
   Section 3 – Timeline: Concrete days/weeks for delivery matching client's deadline (1-2 sentences).
   Section 4 – Price: A specific dollar or local-currency amount aligned with budget (1 sentence).
   Section 5 – Proof Point: Exactly one past project/experience from the verified CV (2-3 sentences).
   Section 6 – Clarifying Question: One smart technical question ending with ?
3. LENGTH IS MANDATORY: body MUST contain 170-230 words (hard floor 150, hard ceiling 250). Aim ~200. Under 150 = REJECTED. Expand Step 1/2/3 and the proof point — do not write terse one-liners.
4. Truthfulness: Never invent facts, companies, or tools. Never claim skills listed as missing.
5. Do NOT include markdown code fences, headers, or any text outside the JSON object.

EXAMPLE STRUCTURE (do not copy verbatim — expand each part to hit ~200 words total):
{
  "kind": "proposal",
  "subject": "Proposal: [Title] - [Name]",
  "body": "I understand you need [problem in client words]. [second sentence on goal/challenge].\\n\\nStep 1: [setup detail]. [extra sentence].\\nStep 2: [core work detail]. [extra sentence].\\nStep 3: [validation detail]. [extra sentence].\\n\\nTimeline: I can deliver within X days.\\n\\nPrice: $X,XXX fixed.\\n\\nProof Point: In [past project], I [achievement with tools].\\n\\nClarifying Question: [question]?"
}"""


def _build_mock_cover_letter(profile: ProfileJSON, posting: PostingJSON, match_result: MatchResult) -> Draft:
    """Deterministic, high-quality cover letter satisfying §7 length and structure for mock mode."""
    company = posting.company_or_client or "your engineering team"
    title = posting.title or "Machine Learning Engineer"
    matched_skills_str = ", ".join(match_result.matched[:3]) if match_result.matched else "Python and machine learning"

    # Gap handling
    gap_clause = ""
    if match_result.missing_must:
        gap_clause = f"While I am actively deepening my hands-on production depth in {match_result.missing_must[0]}, "
    else:
        gap_clause = "With a strong focus on clean code and reliable architecture, "

    body = (
        f"Dear Hiring Team at {company},\n\n"
        f"I am excited to apply for the {title} position. Having followed your recent technical initiatives, "
        f"I admire your team's commitment to building reliable, high-performance data and AI systems. "
        f"With over 2.5 years of experience in data science and AI engineering, I specialize in architecting scalable pipelines "
        f"that bridge cutting-edge research with production workflows.\n\n"
        f"In my previous work at Creativa Innovation Hub, I engineered a Hybrid RAG architecture combining dense vector search "
        f"and BM25 sparse retrieval, implementing automated evaluation guardrails with RAGAS and LangSmith. Furthermore, through "
        f"intensive deep learning training at Neuromatch, I developed production neural network pipelines with PyTorch, "
        f"leveraging modern Transformers and regularized architectures to optimize inference accuracy and execution speed.\n\n"
        f"{gap_clause}my solid foundation in {matched_skills_str} allows me to rapidly master new technologies and "
        f"deliver immediate business impact. I take pride in writing modular, well-tested code that simplifies collaboration across teams.\n\n"
        f"I would welcome the opportunity to discuss how my technical background and problem-solving approach can accelerate "
        f"your current roadmap. Thank you for your time and consideration.\n\n"
        f"Sincerely,\n{profile.name or 'Al-baraa Hegaz'}"
    )

    words = body.split()
    subject = f"Application: {title} — {profile.name}"
    return Draft(
        kind="cover_letter",
        subject=subject,
        body=body,
        claims_ok=True,
        flagged_claims=[],
    )


def _build_mock_email_message(profile: ProfileJSON, posting: PostingJSON, match_result: MatchResult) -> Draft:
    company = posting.company_or_client or "Hiring Team"
    title = posting.title or "the role"
    skills = ", ".join(match_result.matched[:3]) if match_result.matched else "Python and ML"
    body = (
        f"Dear {company},\n\n"
        f"I am writing to apply for the {title} position. With {profile.years_experience or 2.5} years "
        f"in data science and AI engineering, I bring hands-on experience in {skills}.\n\n"
        f"At Creativa Innovation Hub I engineered a Hybrid RAG pipeline with automated evaluation, "
        f"and through Neuromatch I shipped PyTorch training workflows used in production-style notebooks.\n\n"
        f"I would welcome a short conversation about how I can contribute. "
        f"A full cover letter is available on request.\n\n"
        f"Best regards,\n{profile.name or 'Al-baraa Hegaz'}"
    )
    return Draft(
        kind="email_message",
        subject=f"Application: {title} — {profile.name or 'Candidate'}",
        body=body,
        claims_ok=True,
        flagged_claims=[],
    )


def _build_mock_short_bid(profile: ProfileJSON, posting: PostingJSON, match_result: MatchResult) -> Draft:
    title = posting.title or "your project"
    budget = posting.budget.amount if posting.budget else 500
    days = posting.deadline_days or 7
    body = (
        f"Hi — I understand you need help with {title}. "
        f"I can deliver a clean, documented solution using Python and proven evaluation practices.\n\n"
        f"I will: (1) set up the notebook/environment, (2) implement the core pipeline with clear metrics, "
        f"and (3) hand over runnable code with setup notes. "
        f"Timeline about {days} days; proposed budget around ${budget:,.0f}. "
        f"In a past RAG project I shipped reproducible benchmarks with FastAPI tooling.\n\n"
        f"Which GPU or runtime environment should I target?\n\n"
        f"Thanks,\n{profile.name or 'Al-baraa Hegaz'}"
    )
    return Draft(
        kind="short_bid",
        subject=f"Bid: {title} — {profile.name or 'Candidate'}",
        body=body,
        claims_ok=True,
        flagged_claims=[],
    )


def _build_mock_proposal(profile: ProfileJSON, posting: PostingJSON, match_result: MatchResult) -> Draft:
    """Deterministic, high-quality proposal satisfying §7 length and structure for mock mode."""
    title = posting.title or "Technical Project"
    budget_amt = posting.budget.amount if posting.budget else 1500
    timeline_days = posting.deadline_days or 10

    body = (
        f"Hi there,\n\n"
        f"I understand you need a robust, fully documented pipeline for {title}. "
        f"Ensuring reliable data processing and clean, reproducible code is essential to achieving accurate results on schedule.\n\n"
        f"My proposed 3-step approach is:\n"
        f"1. Setup & Preprocessing: Establish environment, configure data loaders, and validate baseline data integrity.\n"
        f"2. Core Implementation: Build and train the model architecture with PyTorch, applying proven evaluation metrics.\n"
        f"3. Validation & Handover: Deliver a clean, runnable Jupyter notebook with full markdown documentation and setup instructions.\n\n"
        f"Timeline & Price: I can deliver the complete, verified solution within {timeline_days} days for a fixed price of ${budget_amt:,.0f}.\n\n"
        f"Relevant Proof Point: In an AI clinical support project, I developed an end-to-end RAG and evaluation workflow using Python, "
        f"FastAPI, and automated evaluation frameworks, delivering reproducible benchmarks ahead of deadline.\n\n"
        f"To ensure exact alignment: do you have a specific GPU environment (e.g. Colab, RunPod, or local CUDA) preferred for execution?\n\n"
        f"Best regards,\n{profile.name or 'Al-baraa Hegaz'}"
    )

    subject = f"Proposal: {title} — {profile.name}"
    return Draft(
        kind="proposal",
        subject=subject,
        body=body,
        claims_ok=True,
        flagged_claims=[],
    )


def _resolve_draft_kind(posting: PostingJSON, draft_kind: Optional[str] = None) -> str:
    """Normalize draft_kind to one of: cover_letter, email_message, proposal, short_bid."""
    is_freelance = posting.type == "project"
    if draft_kind:
        k = draft_kind.strip().lower().replace(" ", "_")
        aliases = {
            "cover": "cover_letter",
            "cover_letter": "cover_letter",
            "email": "email_message",
            "email_message": "email_message",
            "proposal": "proposal",
            "short_bid": "short_bid",
            "bid": "short_bid",
            "short_bid_message": "short_bid",
        }
        resolved = aliases.get(k, k)
        if resolved in ("cover_letter", "email_message", "proposal", "short_bid"):
            return resolved
    return "proposal" if is_freelance else "cover_letter"


def write_draft(
    profile: ProfileJSON,
    posting: PostingJSON,
    match_result: MatchResult,
    client: Optional[LLMClient] = None,
    playbook_retriever: Optional[PlaybookRetriever] = None,
    cv_retriever: Optional[CVRetriever] = None,
    return_trace: bool = False,
    draft_kind: Optional[str] = None,
) -> Any:
    """
    Generates tailored draft (cover letter / email / proposal / short bid) using Playbook and CV RAG.
    Audits claims with claim_checker; auto-regenerates once if unsupported claims are flagged.
    """
    if client is None:
        client = get_client()

    mode = "freelance" if posting.type == "project" else "job"
    is_freelance = (mode == "freelance")
    kind = _resolve_draft_kind(posting, draft_kind)

    if client.mode == "mock":
        builders = {
            "cover_letter": _build_mock_cover_letter,
            "email_message": _build_mock_email_message,
            "proposal": _build_mock_proposal,
            "short_bid": _build_mock_short_bid,
        }
        draft = builders[kind](profile, posting, match_result)
        claims_ok, flagged = check_claims(draft.body, profile, posting, match_result)
        draft.claims_ok = claims_ok
        draft.flagged_claims = flagged
        setattr(draft, "_initial_flagged", list(flagged))
        setattr(draft, "_regenerated", False)
        if return_trace:
            return draft, {
                "initial_flagged": list(flagged),
                "final_flagged": list(flagged),
                "regenerated": False,
                "playbook_rules": [],
                "cv_chunks": [],
                "draft_kind": kind,
            }
        return draft

    # 1. RAG Retrieval
    if playbook_retriever is None:
        playbook_retriever = build_playbook_retriever()
    if cv_retriever is None:
        cv_retriever = build_cv_retriever(profile)

    query_must = " ".join(posting.must_have[:5])
    playbook_rules = playbook_retriever.retrieve(
        query=f"{posting.title} {query_must}",
        mode=mode,
        top_k=3,
    )
    cv_chunks = cv_retriever.retrieve(
        query=f"{posting.title} {query_must}",
        top_k=3,
    )

    # 2. Build Prompt
    system_map = {
        "cover_letter": COVER_LETTER_SYSTEM_PROMPT,
        "email_message": EMAIL_MESSAGE_SYSTEM_PROMPT,
        "proposal": PROPOSAL_SYSTEM_PROMPT,
        "short_bid": SHORT_BID_SYSTEM_PROMPT,
    }
    system_prompt = system_map[kind]
    user_prompt = _construct_writer_prompt(
        profile=profile,
        posting=posting,
        match_result=match_result,
        playbook_rules=playbook_rules,
        cv_chunks=cv_chunks,
        is_freelance=is_freelance,
        draft_kind=kind,
    )

    # 3. Generate Draft
    draft = _call_llm_draft(client, user_prompt, system_prompt, kind)

    # 3b. Length bands per draft kind
    length_bands = {
        "cover_letter": (220, 300, 260),
        "proposal": (150, 250, 200),
        "email_message": (80, 140, 110),
        "short_bid": (60, 120, 90),
    }
    min_w, max_w, target_w = length_bands.get(kind, (150, 250, 200))
    word_count = len(draft.body.split())
    length_regenerated = False
    if word_count < min_w or word_count > max_w:
        logger.info(
            f"Draft length {word_count} outside [{min_w},{max_w}]; regenerating once for length."
        )
        direction = "EXPAND" if word_count < min_w else "SHORTEN"
        length_prompt = (
            f"{user_prompt}\n\n"
            f"CRITICAL LENGTH REVISION: Previous body was {word_count} words "
            f"(required {min_w}-{max_w}). {direction} to about {target_w} words.\n"
            f"PREVIOUS BODY TO REVISE:\n{draft.body}"
        )
        draft = _call_llm_draft(client, length_prompt, system_prompt, kind)
        length_regenerated = True

    # 4. Claim Checking & Single Auto-Regeneration
    claims_ok, flagged = check_claims(draft.body, profile, posting, match_result)
    initial_flagged = list(flagged)
    regenerated = length_regenerated
    if not claims_ok:
        logger.info(f"Draft contained flagged claims: {flagged}. Auto-regenerating once per §7.")
        revision_prompt = (
            f"{user_prompt}\n\n"
            f"CRITICAL REVISION INSTRUCTION: The previous draft was rejected for unverified claims:\n"
            + "\n".join(f"- {f}" for f in flagged) + "\n"
            f"You MUST remove all of the above claims. Only mention tools, numbers, and companies explicitly "
            f"stated in the Verified CV Chunks above. "
            f"Body MUST stay between {min_w} and {max_w} words (aim ~{target_w})."
        )
        draft = _call_llm_draft(client, revision_prompt, system_prompt, kind)
        claims_ok, flagged = check_claims(draft.body, profile, posting, match_result)
        regenerated = True

    # 5. Hard ceiling: deterministic trim if still over max after LLM passes
    if len(draft.body.split()) > max_w:
        draft.body = _trim_body_to_max_words(draft.body, max_w)
        claims_ok, flagged = check_claims(draft.body, profile, posting, match_result)
        draft.claims_ok = claims_ok
        draft.flagged_claims = flagged
        length_regenerated = True
        regenerated = True

    draft.kind = kind  # type: ignore[assignment]
    draft.claims_ok = claims_ok
    draft.flagged_claims = flagged
    setattr(draft, "_initial_flagged", initial_flagged)
    setattr(draft, "_regenerated", regenerated)

    if return_trace:
        return draft, {
            "initial_flagged": initial_flagged,
            "final_flagged": list(flagged),
            "regenerated": regenerated,
            "length_regenerated": length_regenerated,
            "word_count": len(draft.body.split()),
            "playbook_rules": playbook_rules,
            "cv_chunks": cv_chunks,
            "draft_kind": kind,
        }
    return draft


def _construct_writer_prompt(
    profile: ProfileJSON,
    posting: PostingJSON,
    match_result: MatchResult,
    playbook_rules: List[str],
    cv_chunks: List[str],
    is_freelance: bool,
    draft_kind: str = "cover_letter",
) -> str:
    """Builds detailed grounding prompt for the selected draft kind."""
    labels = {
        "cover_letter": ("Cover Letter", "220–300 words", "about 260 words"),
        "proposal": ("Proposal", "150–250 words", "about 200 words"),
        "email_message": ("short application Email message", "80–140 words", "about 110 words"),
        "short_bid": ("short Bid message", "60–120 words", "about 90 words"),
    }
    doc_type, length_guideline, aim = labels.get(
        draft_kind, ("Draft", "100–200 words", "about 150 words")
    )

    prompt = f"""Generate a winning {doc_type} strictly between {length_guideline}.
Set JSON "kind" to "{draft_kind}".

TARGET POSTING:
- Title: {posting.title}
- Organization / Client: {posting.company_or_client or 'Hiring Manager'}
- Type: {posting.type}
- Must-have requirements: {', '.join(posting.must_have) if posting.must_have else 'Standard technical requirements'}
- Summary / Scope: {posting.summary or 'Full execution as described'}
"""
    if posting.budget:
        prompt += f"- Budget: ${posting.budget.amount:,.0f} ({posting.budget.kind})\n"
    if posting.deadline_days:
        prompt += f"- Client Deadline: {posting.deadline_days} days\n"

    prompt += f"""
CANDIDATE QUALIFICATIONS (Ground Truth):
- Name: {profile.name}
- Headline: {profile.headline}
- Experience: {profile.years_experience} years
- Matched Core Skills: {', '.join(match_result.matched) if match_result.matched else 'Python, Data Science'}
- Missing Must-Have Skills (DO NOT CLAIM THESE): {', '.join(match_result.missing_must) if match_result.missing_must else 'None'}
"""
    if match_result.advice:
        prompt += f"- Strategic Advice: {match_result.advice}\n"

    prompt += "\nVERIFIED CV CHUNKS (Use only these facts for achievements and proof points):\n"
    for i, chunk in enumerate(cv_chunks, 1):
        prompt += f"[{i}] {chunk}\n"

    prompt += "\nRELEVANT PLAYBOOK STRATEGY RULES (Apply these techniques):\n"
    for i, rule in enumerate(playbook_rules, 1):
        prompt += f"[{i}] {rule}\n"

    prompt += f"""
STRICT EXECUTION CONSTRAINTS:
1. Target length: strictly {length_guideline} — aim for {aim}.
2. Never claim skills listed in Missing Must-Have Skills ({match_result.missing_must}).
3. Never invent employers, years, or numbers.
4. Return only the JSON object with fields: "kind", "subject", "body".
"""
    if draft_kind == "proposal":
        prompt += (
            "5. Body MUST open with 'I understand you need...' and include "
            "labeled 'Step 1:', 'Step 2:', and 'Step 3:' lines.\n"
        )
    return prompt


def _trim_body_to_max_words(body: str, max_words: int) -> str:
    """
    Deterministic trim when LLM still exceeds the §7 ceiling after one regen.
    Keeps leading/trailing paragraphs (hook/CTA or problem/question) and shortens the middle.
    """
    words = body.split()
    if len(words) <= max_words:
        return body
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    if len(paragraphs) <= 2:
        return " ".join(words[:max_words])
    # Preserve first and last paragraph; shrink middle until under cap
    head, *middle, tail = paragraphs
    budget = max_words - len(head.split()) - len(tail.split())
    if budget < 20:
        return " ".join(words[:max_words])
    mid_text = " ".join(" ".join(middle).split()[:budget])
    return f"{head}\n\n{mid_text}\n\n{tail}"


def _call_llm_draft(client: LLMClient, user_prompt: str, system_prompt: str, kind: str) -> Draft:
    """Invokes LLM and extracts valid Draft schema. `kind` is the draft type string."""
    if isinstance(kind, bool):
        kind = "proposal" if kind else "cover_letter"
    max_tokens = 700 if kind in ("email_message", "short_bid") else 1100
    try:
        response_text = client.generate(
            prompt=user_prompt,
            system=system_prompt,
            temperature=config.TEMP_WRITER,
            max_new_tokens=max_tokens,
        )
        data = extract_json(response_text, DraftContent)
        body = data.body.strip() if data.body else response_text.strip()
        subject = data.subject.strip() if data.subject else f"Application: {kind.replace('_', ' ').title()}"
        out_kind = data.kind if data.kind in (
            "cover_letter", "proposal", "email_message", "short_bid"
        ) else kind
        return Draft(
            kind=out_kind,  # type: ignore[arg-type]
            subject=subject,
            body=body,
            claims_ok=True,
            flagged_claims=[],
        )
    except Exception as e:
        logger.warning(f"Draft generation failed ({e}), constructing structured fallback.")
        return Draft(
            kind=kind,  # type: ignore[arg-type]
            subject=f"Application for {kind}",
            body=f"Dear Team,\n\nI am writing to express my strong interest in this position.\n\nBest regards,\nCandidate",
            claims_ok=True,
            flagged_claims=[],
        )


def refine_draft(
    draft: Draft,
    instruction: str,
    profile: ProfileJSON,
    posting: PostingJSON,
    match_result: Optional[MatchResult] = None,
    client: Optional[LLMClient] = None,
) -> Draft:
    """
    Refines an existing draft per user instruction (shorter, more formal, friendlier, or custom).
    Maintains all §7 truthfulness and length rules.
    """
    if client is None:
        client = get_client()

    inst_lower = instruction.lower().strip()
    instruction_guidance = instruction
    if "shorter" in inst_lower:
        instruction_guidance = "Make the draft significantly more concise while preserving key proof points and required structure."
    elif "formal" in inst_lower:
        instruction_guidance = "Elevate the tone to be more executive, formal, and authoritative, while keeping warmth."
    elif "friendlier" in inst_lower:
        instruction_guidance = "Infuse a warmer, more engaging, highly collaborative and conversational tone."

    if client.mode == "mock":
        # Mock refinement
        refined_body = draft.body + f"\n\n[Refined with instruction: {instruction_guidance}]"
        return Draft(
            kind=draft.kind,
            subject=draft.subject,
            body=refined_body,
            claims_ok=True,
            flagged_claims=[],
        )

    prompt = f"""Please revise the following {draft.kind.replace('_', ' ')} draft based on this instruction:
"{instruction_guidance}"

CURRENT DRAFT:
Subject: {draft.subject}
Body:
{draft.body}

RULES TO PRESERVE:
- Maintain truthful claims only (no invented facts, no claims of missing skills).
- Keep kind="{draft.kind}".
- Return ONLY the JSON object with 'kind', 'subject', 'body'.
"""
    system_map = {
        "cover_letter": COVER_LETTER_SYSTEM_PROMPT,
        "email_message": EMAIL_MESSAGE_SYSTEM_PROMPT,
        "proposal": PROPOSAL_SYSTEM_PROMPT,
        "short_bid": SHORT_BID_SYSTEM_PROMPT,
    }
    system_prompt = system_map.get(draft.kind, COVER_LETTER_SYSTEM_PROMPT)
    refined = _call_llm_draft(client, prompt, system_prompt, draft.kind)

    claims_ok, flagged = check_claims(refined.body, profile, posting, match_result)
    refined.kind = draft.kind
    refined.claims_ok = claims_ok
    refined.flagged_claims = flagged
    return refined
