# JSBrother — Architecture & Milestone Handoff

> Multi-agent AI assistant for **job seekers** (cover letters) and **freelancers** (project proposals).
> It scores the fit, tells the user **APPLY / CAUTION / SKIP**, writes the draft, and (with approval) sends it and logs it.

---

## 0. START HERE (every agent / model / account reads this first)

**Prompt to give a new agent:** "Read ARCHITECTURE.md fully. Continue from the first unchecked milestone in §11. Follow the rules in §0."

**Rules**
1. Find the first milestone in §11 that has unchecked boxes. Work only on it.
2. Tick `- [x]` the moment a task is truly done and tested. Never tick untested work.
3. Before stopping (quota/limit), add a row to §14 Handoff Log: what is done, what is half-done, exact next step.
4. Do not change a locked decision (§2) without adding a row to §13 Decision Log with the reason.
5. Keep the folder structure of §10. Keep files small and one-purpose.
6. Never hard-code secrets. Use `.env` (see `.env.example`). Never commit `.env`.
7. Always keep `LLM_MODE=mock` working, so development continues without the GPU endpoint.

**Current milestone:** `M9`  ← update this line when you finish a milestone
**Last updated by:** Composer / 2026-10-22

---

## 1. What JSBrother does

| | Job Seeker mode | Freelancer mode |
|---|---|---|
| Input | CV (PDF) + job posting (pasted text, URL optional) | Skills profile (PDF/text) + project post (pasted text) |
| Analysis | Match score, strengths, gaps | Match score, strengths, gaps, budget fit, deadline fit, red flags |
| Verdict | APPLY / CAUTION / SKIP | APPLY / CAUTION / SKIP |
| Draft | Tailored **cover letter** | Tailored **proposal** (problem, approach, timeline, price) |
| Action | Email application (after approval) + log | Download/copy proposal + log (optional email) |

**Output order in the GUI:** (1) Verdict card → (2) editable draft → (3) Approve & Send.
**If SKIP:** only the card is shown, with an **"Apply anyway"** button that generates the draft.

---

## 2. Locked decisions

| Topic | Decision |
|---|---|
| Name | JSBrother (Job Seekers' Brother) |
| LLM | Qwen2.5-7B-Instruct, 4-bit (bitsandbytes), served on Kaggle/Colab GPU |
| LLM access | FastAPI `/generate` + ngrok, Bearer key. Streamlit app calls it over HTTP |
| Orchestration | LangChain LCEL (`RunnableLambda`, `RunnableParallel`, `\|`). No LangGraph |
| Retrieval | `sentence-transformers/all-MiniLM-L6-v2` + FAISS (`IndexFlatL2` on normalized vectors) |
| Who decides | **Python** computes score and verdict. LLM only extracts, explains, and writes |
| Memory | Buffer memory (full session) + persisted `profile.json` |
| GUI | Streamlit, dark theme "Midnight Aurora" |
| Email | Gmail SMTP + app password, **dry-run ON by default**, human approval required |
| Tracker | `data/tracker.csv` (Google Sheet via gspread = stretch) |
| Language | Python 3.10+ |

---

## 3. Architecture

```
┌──────────────────────── Streamlit GUI (app.py) ────────────────────────┐
│ sidebar: mode · reset │ Analyze tab │ Tracker tab │ Settings tab       │
└───────────────┬─────────────────────────────────────────────────────────┘
                │ question + files
                ▼
        ┌── LangChain LCEL pipeline (orchestrator.py) ──┐
        │ 1 Router            detect mode, validate input│
        │ 2 ┌ Profile Analyzer ┐  (RunnableParallel)     │
        │   └ Posting Analyzer ┘                         │
        │ 3 Matcher           score + gaps + VERDICT     │
        │ 4 Gate              SKIP? stop (unless forced) │
        │ 5 Writer            playbook RAG + CV RAG      │
        │ 6 Claim Checker     no invented facts          │
        │ 7 Remember          buffer memory + trace      │
        └───────────────┬────────────────────────────────┘
                        ▼ (user clicks Approve & Send)
                 Action Agent: email + tracker + save draft

        LLM server (Kaggle/Colab GPU) ◄── HTTP ── llm_client.py
```

Parallel step: Profile and Posting analyzers run concurrently (thread pool). The single GPU endpoint may queue them; the design is still parallel.

---

## 4. Agents and their tools

| Agent | Job | Tools | Uses LLM? |
|---|---|---|---|
| **Router** | Detects mode (job/project), checks inputs, starts pipeline | keyword rules + short LLM fallback, LCEL runner | rarely |
| **Profile Analyzer** | CV/profile → `ProfileJSON` | `pdf_reader` (pypdf), `llm_client`, `json_utils` (retry) | yes (extract) |
| **Posting Analyzer** | Posting → `PostingJSON` | `web_fetcher` (requests + BeautifulSoup, best-effort), `llm_client`, `json_utils` | yes (extract) |
| **Matcher** | Score, gaps, verdict, advice | `skills.py` (alias map + MiniLM similarity), `compute_score`, `decide_verdict` | only to phrase reasons |
| **Writer** | Cover letter / proposal | Playbook retriever (FAISS), CV-chunk retriever (FAISS), `llm_client`, `claim_checker` | yes (draft) |
| **Action Agent** | Send + log | `email_sender` (smtplib), `tracker` (CSV), `save_draft` | no |

All tools are LangChain `StructuredTool`s with typed args and docstrings, called with `.invoke({...})`.

---

## 5. Data schemas (`jsb/schemas.py`)

```json
ProfileJSON = {
  "name": "", "email": "", "headline": "",
  "skills": ["python", "faiss"],
  "years_experience": 2.5,
  "roles": [{"title": "", "company": "", "years": 1.0, "highlights": [""]}],
  "education": [""], "projects": [""], "languages": [""], "links": [""],
  "min_budget": null, "min_lead_days": 7      // freelancer settings from GUI
}

PostingJSON = {
  "title": "", "company_or_client": "", "type": "job|project",
  "must_have": [""], "nice_to_have": [""],
  "min_years": null, "deal_breakers": [""],
  "budget": {"amount": 0, "currency": "USD", "kind": "fixed|hourly"} | null,
  "deadline_days": null, "vague_scope": false,
  "summary": "", "contact_email": null
}

MatchResult = {
  "score": 0-100, "must_coverage": 0-1, "nice_coverage": 0-1,
  "matched": [""], "missing_must": [""], "missing_nice": [""],
  "deal_breaker_hits": [""], "verdict": "APPLY|CAUTION|SKIP",
  "reasons": ["", "", ""], "advice": "",
  "freelancer": {"budget_fit": 0-1, "deadline_fit": 0-1, "red_flags": [""]} | null
}

Draft = {
  "kind": "cover_letter|proposal", "subject": "", "body": "",
  "claims_ok": true, "flagged_claims": [""]
}
```

All LLM JSON goes through `json_utils.extract_json()`: strip code fences, find outer `{}`, validate against schema, **retry up to 2 times** with a "return only valid JSON" reminder. On final failure return a clear error to the GUI (never crash).

---

## 6. Matching and verdict rules (`matcher.py`) — Python only

**Skill match:** normalize (lowercase, trim), alias map (`js→javascript`, `ml→machine learning`, …), then exact/alias match, else MiniLM cosine similarity ≥ `SKILL_SIM_THRESHOLD` (default 0.70, in `config.py`).

**Score**
- Job: `100 × (0.60·must_cov + 0.20·nice_cov + 0.20·exp_fit)`
  `exp_fit = min(1, user_years / min_years)`, or 1 if no `min_years`.
- Freelancer: `100 × (0.55·must_cov + 0.15·nice_cov + 0.15·budget_fit + 0.15·deadline_fit)`
  - `budget_fit`: 1 if budget ≥ user's `min_budget`; 0.5 if within 20% below; 0 if lower; 0.7 if budget unknown.
  - `deadline_fit`: 1 if no deadline or `deadline_days ≥ min_lead_days`; 0.5 if ≥ half of it; else 0.

**Verdict (apply in this exact order)**
1. any `deal_breaker_hits` → **SKIP**
2. `score < 50` or `len(missing_must) ≥ 3` → **SKIP**
3. `score ≥ 75` and `len(missing_must) == 0` → **APPLY**
4. otherwise → **CAUTION**
5. Freelancer only: if `vague_scope` or `budget_fit == 0`, downgrade APPLY → CAUTION and add a red flag.

The LLM never changes score or verdict. It only phrases `reasons` (3 bullets) and `advice` (1 line) from the computed facts.

---

## 7. Writer (`writer.py`)

**Inputs:** `ProfileJSON`, `PostingJSON`, `MatchResult`, top-3 playbook chunks, top-3 CV chunks (retrieved by the posting's must-haves).

**Cover letter (≈220–300 words):** hook tied to the company/role → 2–3 achievements that match must-haves (from CV only) → honest, brief handling of a gap ("eager to deepen X") → clear call to action.

**Proposal (≈150–250 words):** restate the client's problem in their words → approach in 3 steps → timeline → price → one relevant proof point → one clarifying question.

**Rules:** never claim a skill listed in `missing_must`; no invented numbers, employers, or years; no filler openers.

**Playbook (user's 50 proposal-tips slides):**
- Drop the slides into `data/playbook/slides/`.
- M5 extracts the text and distils it into `data/playbook/playbook.md` (≤ 40 short rules, each tagged `cover_letter`, `proposal`, or `both`).
- Rules are chunked, embedded, and stored in a FAISS index. The Writer retrieves the closest rules and puts them in the prompt.
- If no slides exist yet, the built-in rules above are used.

**Claim Checker (`claim_checker.py`):** extract numbers, years, tool names, and company names from the draft; each must appear in the profile text or posting text. Unsupported claims → regenerate once with "remove unsupported claims"; if still flagged, show them as a warning in the GUI.

**Refine (follow-ups):** quick buttons (shorter / more formal / friendlier) and a free-text box. The Writer re-runs using buffer memory (the previous draft and the user's instruction).

---

## 8. Memory (`memory.py`)

- `BufferMemory`: full session list of `{user, assistant}` turns, with `reset()`.
- Persisted `data/profile.json` so the user doesn't re-upload the CV for every posting.
- `last_posting` and `last_draft` kept for follow-ups.
- Optional 3-line summary every 4 turns (LLM).
- Streamlit reruns the script on every click, so session data lives in `st.session_state`.

---

## 9. Streamlit GUI (`app.py`)

**Layout**
- **Sidebar:** mode switch (Job Seeker / Freelancer), freelancer settings (min budget, min lead days), dry-run toggle, Reset.
- **Analyze tab:** (1) upload CV/profile (or reuse saved) → (2) paste posting (+ optional URL) → **Analyze** button → verdict card → editable draft → Refine buttons → **Approve & Send** (or Download for freelancers).
- **Tracker tab:** table of past applications (from CSV) with filters and a download button.
- **Settings tab:** LLM URL + key (the ngrok URL changes every restart), SMTP email + app password, test-connection buttons.

**Verdict card:** circular score gauge (inline SVG, colored by verdict), verdict badge, 3 reasons, matched skills as green chips, missing skills as red-outline chips, one-line advice. Freelancer mode adds budget/deadline indicators and red flags.

**Theme "Midnight Aurora"** (`.streamlit/config.toml` + a small CSS block):

| Role | Value |
|---|---|
| base | dark |
| Background | `#0F1220` |
| Sidebar / secondary bg | `#161A2E` |
| Cards | `#1C2138`, radius 16px, soft shadow |
| Text | `#E8EAF6` |
| Primary | `#6C5CE7` |
| Header + buttons | gradient `#6C5CE7 → #00B8D9` |
| Highlights | cyan glow `#00B8D9` |
| APPLY / CAUTION / SKIP | `#2E9E5B` / `#F2A33A` / `#D64545` |
| Font | Inter (Google Fonts import in CSS) |

Mode accent: Job Seeker badge cyan `#00B8D9`, Freelancer badge violet `#6C5CE7`. Keep gradients subtle for comfort.

**UX rules:** show a spinner and a step-by-step progress ("Reading CV… Matching… Writing…"); clear error messages (endpoint down, bad PDF, empty posting); nothing is emailed without the explicit Approve click.

---

## 10. Folder structure

```
jsbrother/
├── ARCHITECTURE.md          ← this file
├── README.md
├── .env.example             LLM_MODE, LLM_BASE_URL, LLM_API_KEY, SMTP_USER, SMTP_APP_PASSWORD
├── requirements.txt
├── .streamlit/config.toml
├── app.py                   Streamlit entry
├── config.py                thresholds, weights, paths
├── jsb/
│   ├── llm_client.py        HTTP client + retries + mock mode
│   ├── schemas.py
│   ├── json_utils.py
│   ├── skills.py            normalize, aliases, similarity
│   ├── memory.py
│   ├── orchestrator.py      LCEL pipeline
│   ├── agents/              router, profile_analyzer, posting_analyzer, matcher, writer, action_agent
│   └── tools/               pdf_reader, web_fetcher, vector_store, claim_checker, email_sender, tracker
├── data/
│   ├── playbook/ (slides/, playbook.md)
│   ├── samples/ (cvs/, postings/)
│   ├── drafts/   tracker.csv   profile.json
├── llm_server/kaggle_server.ipynb
└── tests/ (test_json_utils, test_matcher, test_claim_checker, test_pipeline, eval/)
```

---

## 11. Milestones (tick as you go)

### M0 — Setup and LLM endpoint
- [x] Create folder structure, `requirements.txt`, `.env.example`, `config.py`
- [x] `llm_server/kaggle_server.ipynb` from the existing NGROK notebook **with 3 fixes**: (a) decode only new tokens, not the prompt; (b) API key and ngrok token from secrets/env, not hard-coded; (c) Qwen2.5-7B-Instruct 4-bit using the chat template; add `GET /health`
- [x] `llm_client.py`: `generate(prompt, system, max_new_tokens, temperature)`, timeout, 2 retries, `LLM_MODE=mock|remote`
- [x] Mock mode returns canned valid JSON/text for each agent
- **Done when:** `generate("hi")` works in mock and remote modes.

### M1 — Schemas and JSON utils
- [x] `schemas.py` (dataclasses or pydantic) for the 4 objects in §5
- [x] `json_utils.extract_json()` with fence stripping, validation, 2 retries
- [x] `tests/test_json_utils.py` (clean JSON, fenced JSON, broken JSON)
- **Done when:** tests pass.

### M2 — Profile Analyzer
- [x] `pdf_reader.py`; paste-text fallback
- [x] `profile_analyzer.py` → valid `ProfileJSON`; save to `data/profile.json`
- [x] 3 sample CVs in `data/samples/cvs/` with hand-labeled skills
- **Done when:** ≥ 80% of labeled skills extracted on all 3 CVs.

### M3 — Posting Analyzer
- [x] `web_fetcher.py` (best-effort; many sites block scraping, so pasted text is the primary path)
- [x] `posting_analyzer.py` → valid `PostingJSON` (must/nice split, deal-breakers, budget, deadline, vague_scope)
- [x] 6 sample postings (3 job, 3 project) in `data/samples/postings/`
- **Done when:** all 6 give valid JSON; must-haves match the hand labels.

### M4 — Matcher and verdict
- [x] `skills.py` (aliases + MiniLM similarity + threshold from config)
- [x] `matcher.py`: score formulas and verdict rules from §6 (pure Python)
- [x] `tests/test_matcher.py` covering each verdict rule (deal-breaker, <50, ≥3 missing, APPLY, CAUTION, freelancer downgrade)
- [x] LLM phrases `reasons` and `advice` only
- **Done when:** all rule tests pass; the LLM cannot change the verdict.

### M5 — Playbook RAG and Writer
- [x] Ingest user's slides → `playbook.md` (≤ 40 tagged rules) *(skip to built-in rules if slides not attached yet; revisit later)*
- [x] `vector_store.py`: build and query FAISS for playbook and for CV chunks
- [x] `writer.py`: cover letter and proposal prompts, per §7
- [x] `claim_checker.py` + one auto-regenerate
- [x] Refine function (shorter / formal / friendly / free text)
- **Done when:** 5 drafts have no flagged claims and follow the length and structure rules. ✅ (logs/m5_eval_drafts_v3.txt)

### M6 — Memory and orchestrator
- [x] `memory.py` (buffer, persisted profile, last posting/draft)
- [x] `orchestrator.py`: LCEL chain (Router → parallel analyzers → Matcher → Gate → Writer → Claim Checker → Remember)
- [x] Router mode detection; SKIP gate with `force=True` option
- [x] Decision trace printed/returned (steps, timings, retrieved chunks)
- **Done when:** `run(profile, posting, mode)` returns verdict + draft end to end in mock and remote modes. ✅

### M7 — Actions
- [x] `email_sender.py` (Gmail SMTP SSL, dry-run default, optional CV attachment)
- [x] `tracker.py` (append/read CSV: timestamp, mode, title, company, score, verdict, draft_path, email_status, recipient)
- [x] `save_draft()` to `data/drafts/`
- [x] Freelancer path: download/copy + log (email optional)
- **Done when:** a dry-run send and one real test email both log correctly. ✅

### M8 — Streamlit GUI
- [x] `.streamlit/config.toml` + CSS (Midnight Aurora, §9)
- [x] Sidebar, Analyze tab, Tracker tab, Settings tab
- [x] Verdict card (SVG gauge, chips, reasons, advice) and SKIP + "Apply anyway"
- [x] Editable draft, Refine buttons, Approve & Send
- [x] Progress steps, spinners, friendly errors; state in `st.session_state`
- **Done when:** a non-technical user completes both modes without help. ✅

### M9 — Testing and evaluation
- [ ] Eval set: 8–10 job pairs + 6–8 project pairs with expected verdicts
- [ ] `tests/eval/run_eval.py` prints a results table (see §12)
- [ ] Fix weak spots; record final numbers in README
- **Done when:** the table meets the targets or the gaps are documented.

### M10 — Docs and video
- [ ] README (setup, run, screenshots), architecture diagram
- [ ] Video per §15 (face-cam, live demo + explanation)
- **Done when:** a fresh clone runs from the README alone.

### M11 — Stretch (only if M0–M10 are done)
- [ ] LoRA fine-tune on proposal examples (Week 2 fine-tuning)
- [ ] Google Sheet tracker via gspread
- [ ] Batch mode: score several postings in parallel, ranked list

---

## 12. Evaluation

| Metric | How | Target |
|---|---|---|
| Extraction accuracy | skills extracted vs hand-labeled | ≥ 80% |
| Verdict accuracy | verdict vs expected on the eval set | ≥ 85% |
| Score sanity | score inside the expected band | ≥ 85% |
| Hallucination | flagged claims in final drafts | 0 |
| JSON validity | valid on first try / after retry | report both |
| Action success | email (dry-run + real) sent and row logged | 100% |
| Efficiency | end-to-end seconds per run | report mean |
| Reliability | runs finished without crash | ≥ 95% |

---

## 13. Decision Log

| Date | Decision | Reason | By |
|---|---|---|---|
| — | Python decides verdict, LLM only explains | Consistency; small model can't contradict the rules | design |
| 2026-10-21 | LLM server runs on Colab T4 instead of Kaggle | Kaggle GPU option was unavailable; the notebook already supports Colab secrets | user |

---

## 14. Handoff Log (append a row when you stop)

| Date | Agent/account | Milestone | Done | Half-done | Next exact step |
|---|---|---|---|---|---|
| — | — | — | — | — | Start M0 |
| 2026-10-21 | Gemini 3.8 Flash | M0 | Setup, folder structure, config, server notebook, llm_client with mock & remote modes tested (10/10 tests passed) | None | Start M1: Schemas and JSON utils |
| 2026-10-21 | Gemini 3.8 Flash | M1 | Pydantic v2 schemas for all 4 models and json_utils with fence stripping, repair & retries (22/22 total tests passed) | None | Start M2: Profile Analyzer (pdf_reader, profile_analyzer, sample CVs) |
| 2026-10-21 | Gemini 3.8 Flash | M2 | `pdf_reader.py` (Unicode handling + paste fallback), `profile_analyzer.py` (saves `data/profile.json`), 2 synthetic CVs + `labels.json`. Extraction evaluated on remote Colab LLM across all 3 CVs (CORE recall: 96.6%, 100%, 100% — all ≥80%). 26/26 tests passed. | None | Start M3: Posting Analyzer (`web_fetcher.py`, `posting_analyzer.py`, 6 sample postings in `data/samples/postings/`) |
| 2026-10-21 | Claude Sonnet | M3 | `web_fetcher.py` (URL fetch + pasted text + local file), `posting_analyzer.py` (valid PostingJSON), 6 synthetic postings + `labels.json`, `tests/test_posting_analyzer.py` (7 tests). Remote eval: all 6 synthetic MUST recall ≥80% (100%, 100%, 87.5%, 100%, 100%, 100%); all JSON valid. Real postings: real_job_2=92.9%, real_job_1/3=76.9%, real_project_1=66.7%, real_project_2=55.6%. Fixed null-coercion in schemas.py (field_validators). 33/33 tests passed. | None | Start M4: Matcher (`skills.py`, `matcher.py`, `tests/test_matcher.py`) |
| 2026-10-22 | Gemini 3.8 Flash | M4 | Fixed deal-breaker logic: only demonstrable failures (e.g. required years > profile years, missing required tool months) trigger hits; satisfied requirements pass; unverifiable conditions (NDA, work authorization, payment terms, deadlines, scraper examples) surface as warnings/red flags without auto-SKIP. Fixed skill matching: pipe alternatives evaluated individually, candidate skills enriched from roles/projects, and expanded tech aliases (deep learning, vision, transfer learning, math, databases). 46/46 pytest tests pass. Evaluated on 11 postings: real_job_1=APPLY (80.0), real_project_1=CAUTION (95.0), real_project_2=CAUTION (82.5), real_job_3=SKIP (45.6). | None | Start M5: Playbook RAG and Writer (`playbook.md`, `vector_store.py`, `writer.py`, `claim_checker.py`) |
| 2026-10-22 | Composer | M5–M7 | M5: length-regen + deterministic trim; 5/5 drafts PASS (0 flagged, length+structure OK) in `logs/m5_eval_drafts_v3.txt`. Kept built-in `playbook.md`. M6: `memory.py`, `router.py`, LCEL `orchestrator.py` with SKIP gate/`force`, decision trace; mock+remote `run()` OK. M7: SMTP dry-run + real test email sent; tracker CSV + `save_draft` + freelancer path. | None | Start M8: Streamlit GUI (Midnight Aurora) |
| 2026-10-22 | Composer | M8 | Midnight Aurora GUI: `app.py`, `assets/style.css`, `.streamlit/config.toml`, `jsb/gui_helpers.py`. Sidebar (mode/mock/dry-run/freelancer settings/Reset), Analyze/Tracker/Settings, SVG gauge+chips, SKIP Apply anyway, refine, Approve&Send/Download. Tests 7/7; mock flows job+freelance dry-run; Streamlit boots :8501. Installed streamlit into `.venv`. | None | Start M9: eval set + `tests/eval/run_eval.py` |

---

## 15. Training mapping and video

**Training mapping**

| Week | Applied in JSBrother |
|---|---|
| 1 LLMs & Agents | Qwen model; six agents with role, tools, goal |
| 2 RAG & Fine-tuning | FAISS over playbook and CV; optional LoRA (M11) |
| 3 LangChain & Memory | LCEL pipeline, StructuredTools, buffer memory, follow-up refine |
| 4 Multi-agent & Automation | Router, parallel analyzers, email + tracker automation |
| 5 GUI | Streamlit app with Midnight Aurora theme |

**Final-project objectives**

| Objective | Where |
|---|---|
| 1 Data retrieval | CV/profile PDFs, posting text/URL, playbook, tracker CSV |
| 2 Processing and summarization | Analyzers, Matcher reasons, Writer |
| 3 Automated actions | Email, tracker log, saved drafts |
| 4 Workflow orchestration | LCEL pipeline with parallel step and gate |
| 5 Presentation | README + face-cam video |
| 6 Testing and evaluation | §12 table |

**5-minute video outline:** 0:00 problem (applying is slow, generic) → 0:40 two modes → 1:10 architecture and agents → 2:00 RAG + playbook + claim checker → 2:45 live demo, job seeker (verdict + cover letter + send) → 3:30 live demo, freelancer (match + proposal) → 4:15 evaluation table → 4:45 close.

---

## 16. Known pitfalls

- ngrok URL changes every restart → paste the new URL in Settings.
- Kaggle/Colab sessions time out → keep `LLM_MODE=mock` for GUI and logic work.
- JSON drift from the 7B model → temperature 0 for extraction, retries, validation.
- Writer temperature 0.4–0.7; extraction and explanation stay near 0.
- LinkedIn/Upwork usually block scraping → pasted text is the main path; the URL fetch is a bonus.
- Gmail needs 2FA + an app password; never commit credentials.
- Streamlit reruns on every click → keep state in `st.session_state`.
- Truncate long CVs/postings to fit the context window (chunk first, then summarize).
- Privacy: CV data stays local; don't log full CV text in traces.
