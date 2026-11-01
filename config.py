"""
JSBrother — Configuration & Constants
Thresholds, weights, paths, and environment settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if python-dotenv is available
BASE_DIR = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env", override=True)
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
PROFILE_PATH = DATA_DIR / "profile.json"
TRACKER_PATH = DATA_DIR / "tracker.csv"
DRAFTS_DIR = DATA_DIR / "drafts"
PLAYBOOK_DIR = DATA_DIR / "playbook"
PLAYBOOK_FILE = PLAYBOOK_DIR / "playbook.md"
PLAYBOOK_SLIDES_DIR = PLAYBOOK_DIR / "slides"
SAMPLES_DIR = DATA_DIR / "samples"
SAMPLES_CVS_DIR = SAMPLES_DIR / "cvs"
SAMPLES_POSTINGS_DIR = SAMPLES_DIR / "postings"

# Ensure essential runtime folders exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
DRAFTS_DIR.mkdir(parents=True, exist_ok=True)
PLAYBOOK_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LLM & Server Settings
# -----------------------------------------------------------------------------
LLM_MODE = os.getenv("LLM_MODE", "mock").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000").rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LLM_TIMEOUT = 180
DEFAULT_MAX_RETRIES = 2

# Generation temperatures per role
TEMP_EXTRACTION = 0.0
TEMP_EXPLANATION = 0.0
TEMP_WRITER = 0.5

# -----------------------------------------------------------------------------
# Embeddings & Vector Store
# -----------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SKILL_SIM_THRESHOLD = 0.70

# -----------------------------------------------------------------------------
# Matcher Weights & Thresholds
# -----------------------------------------------------------------------------
# Job Seeker Score: 100 * (0.60 * must + 0.20 * nice + 0.20 * exp)
JOB_MUST_WEIGHT = 0.60
JOB_NICE_WEIGHT = 0.20
JOB_EXP_WEIGHT = 0.20

# Freelancer Score: 100 * (0.55 * must + 0.15 * nice + 0.15 * budget + 0.15 * deadline)
FL_MUST_WEIGHT = 0.55
FL_NICE_WEIGHT = 0.15
FL_BUDGET_WEIGHT = 0.15
FL_DEADLINE_WEIGHT = 0.15

# Verdict Decision Thresholds (evaluated in exact order)
VERDICT_SKIP_SCORE_THRESHOLD = 50.0
VERDICT_SKIP_MISSING_MUST_COUNT = 3
VERDICT_APPLY_SCORE_THRESHOLD = 75.0
VERDICT_APPLY_MISSING_MUST_COUNT = 0

# Freelancer Defaults
DEFAULT_MIN_LEAD_DAYS = 7
DEFAULT_MIN_BUDGET = None

# -----------------------------------------------------------------------------
# Email & SMTP Settings
# -----------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_APP_PASSWORD = os.getenv("SMTP_APP_PASSWORD", "")
EMAIL_DRY_RUN = os.getenv("EMAIL_DRY_RUN", "true").strip().lower() == "true"

# -----------------------------------------------------------------------------
# Theme "Midnight Aurora" Design Tokens
# -----------------------------------------------------------------------------
THEME_COLORS = {
    "background": "#0F1220",
    "sidebar": "#161A2E",
    "card": "#1C2138",
    "text": "#E8EAF6",
    "primary": "#6C5CE7",
    "cyan_glow": "#00B8D9",
    "verdict_apply": "#2E9E5B",
    "verdict_caution": "#F2A33A",
    "verdict_skip": "#D64545",
}
