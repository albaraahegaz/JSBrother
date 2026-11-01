"""JSBrother tools package."""

from jsb.tools.pdf_reader import pdf_reader_tool, extract_text_from_pdf, read_pdf_or_text
from jsb.tools.web_fetcher import web_fetcher_tool, fetch_posting_or_text, fetch_url_text
from jsb.tools.vector_store import FAISSIndex, PlaybookRetriever, CVRetriever, build_playbook_retriever, build_cv_retriever
from jsb.tools.claim_checker import check_claims
from jsb.tools.email_sender import send_email, EmailResult
from jsb.tools.tracker import append_tracker_row, read_tracker, TRACKER_FIELDS

__all__ = [
    "pdf_reader_tool",
    "extract_text_from_pdf",
    "read_pdf_or_text",
    "web_fetcher_tool",
    "fetch_posting_or_text",
    "fetch_url_text",
    "FAISSIndex",
    "PlaybookRetriever",
    "CVRetriever",
    "build_playbook_retriever",
    "build_cv_retriever",
    "check_claims",
    "send_email",
    "EmailResult",
    "append_tracker_row",
    "read_tracker",
    "TRACKER_FIELDS",
]
