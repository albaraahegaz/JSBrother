"""
JSBrother — LLM Client
HTTP client for remote Qwen2.5-7B-Instruct endpoint (FastAPI + ngrok) with
automatic retries, timeout handling, and a full mock mode for local offline development.
-> Inspects prompt keywords to return realistic, schema-valid JSON for local testing
generate function:
can run it 100% offline out-of-the-box in mock mode.
"""

import json
import logging
import time
from typing import Optional, Dict, Any
import requests

import config

logger = logging.getLogger("jsbrother.llm_client")


class LLMClient:
    """Client for interacting with the Qwen2.5 LLM endpoint or local mock engine."""

    def __init__(
        self,
        mode: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
    ):
        self.mode = (mode or config.LLM_MODE or "mock").strip().lower()
        self.base_url = (base_url or config.LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or config.LLM_API_KEY
        self.timeout = timeout if timeout is not None else config.DEFAULT_LLM_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else config.DEFAULT_MAX_RETRIES

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate completion from LLM.
        Dispatches to mock generator or remote HTTP endpoint based on self.mode.
        """
        if self.mode == "mock":
            return self._mock_generate(prompt, system)
        return self._remote_generate(prompt, system, max_new_tokens, temperature)

    def check_health(self) -> Dict[str, Any]:
        """Check status of remote or mock server."""
        if self.mode == "mock":
            return {"status": "ok", "mode": "mock", "model": "mock-qwen2.5"}

        url = f"{self.base_url}/health"
        headers = {"ngrok-skip-browser-warning": "true"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _remote_generate(
        self,
        prompt: str,
        system: Optional[str],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Call remote FastAPI /generate endpoint with retries."""
        url = f"{self.base_url}/generate"
        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "prompt": prompt,
            "system": system,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("text", "")
                
                # Server error; retry
                last_error = RuntimeError(
                    f"Remote LLM server error {response.status_code}: {response.text}"
                )
            except (requests.RequestException, Exception) as exc: # Retries network hiccups with 2^attempt sleep, preventing transient drops.
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(1.0 * (attempt + 1))

        raise RuntimeError(
            f"Failed to communicate with remote LLM at {url} after {self.max_retries + 1} attempts: {last_error}"
        )

    def _mock_generate(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Returns realistic, valid canned JSON or text for all JSBrother agents.
        Enables full pipeline execution and unit tests without a GPU.
        """
        prompt_lower = (prompt + " " + (system or "")).lower()

        # 1. Profile Analyzer (CV extraction)
        if "profilejson" in prompt_lower or "extract cv" in prompt_lower or "cv text" in prompt_lower:
            return json.dumps({
                "name": "Alex Mercer",
                "email": "alex.mercer@example.com",
                "headline": "Senior AI & Full-Stack Engineer",
                "skills": ["python", "langchain", "faiss", "fastapi", "docker", "sql", "git"],
                "years_experience": 3.5,
                "roles": [
                    {
                        "title": "Senior AI Developer",
                        "company": "Innovatech Solutions",
                        "years": 2.0,
                        "highlights": [
                            "Built LCEL orchestration pipelines with FAISS vector search",
                            "Engineered production FastAPI services serving 100k requests/day"
                        ]
                    },
                    {
                        "title": "Python Developer",
                        "company": "DataCorp",
                        "years": 1.5,
                        "highlights": [
                            "Automated ETL pipelines using Python and SQL",
                            "Containerized backend services with Docker"
                        ]
                    }
                ],
                "education": ["B.S. in Computer Science, University of Technology, 2021"],
                "projects": [
                    "JSBrother: Autonomous multi-agent job application assistant",
                    "DocuSearch: Retrieval-augmented document analysis tool"
                ],
                "languages": ["English (Native)", "French (Conversational)"],
                "links": ["https://github.com/alexmercer", "https://linkedin.com/in/alexmercer"],
                "min_budget": None,
                "min_lead_days": 7
            }, indent=2)

        # 2. Posting Analyzer (Job or Freelance Project posting)
        if "postingjson" in prompt_lower or "job posting" in prompt_lower or "project posting" in prompt_lower:
            is_project = "freelance" in prompt_lower or "project" in prompt_lower or "proposal" in prompt_lower
            if is_project:
                return json.dumps({
                    "title": "Build LangChain LCEL Multi-Agent Prototype",
                    "company_or_client": "Acuity Ventures",
                    "type": "project",
                    "must_have": ["python", "langchain", "fastapi"],
                    "nice_to_have": ["faiss", "streamlit"],
                    "min_years": None,
                    "deal_breakers": [],
                    "budget": {"amount": 1500, "currency": "USD", "kind": "fixed"},
                    "deadline_days": 14,
                    "vague_scope": False,
                    "summary": "Looking for an expert to build an automated LCEL multi-agent pipeline with Streamlit UI.",
                    "contact_email": "proposals@acuityventures.io"
                }, indent=2)

            return json.dumps({
                "title": "AI Application Developer",
                "company_or_client": "NextGen Systems",
                "type": "job",
                "must_have": ["python", "langchain", "fastapi"],
                "nice_to_have": ["faiss", "docker"],
                "min_years": 2.0,
                "deal_breakers": [],
                "budget": None,
                "deadline_days": None,
                "vague_scope": False,
                "summary": "Seeking an AI engineer skilled in Python and LangChain to develop production agent pipelines.",
                "contact_email": "careers@nextgensystems.com"
            }, indent=2)

        # 3. Matcher Agent (Reasons & Advice phrasing)
        if "matcher" in prompt_lower or "reasons" in prompt_lower or "advice" in prompt_lower:
            return json.dumps({
                "reasons": [
                    "Strong alignment with all must-have skills (Python, LangChain, and FastAPI).",
                    "Demonstrated real-world experience building LCEL pipelines and vector search systems.",
                    "Exceeds minimum experience threshold with 3.5 years of industry practice."
                ],
                "advice": "Emphasize your hands-on production deployment of LangChain LCEL architectures in your application."
            }, indent=2)

        # 4. Claim Checker
        if "claim" in prompt_lower or "hallucination" in prompt_lower:
            return json.dumps({
                "claims_ok": True,
                "flagged_claims": []
            }, indent=2)

        # 5. Router Agent (Mode detection)
        if "router" in prompt_lower or "detect mode" in prompt_lower:
            if "freelance" in prompt_lower or "project" in prompt_lower or "budget" in prompt_lower:
                return "project"
            return "job"

        # 6. Writer Agent (Cover letter or Proposal)
        if "cover letter" in prompt_lower:
            return (
                "Dear Hiring Team at NextGen Systems,\n\n"
                "I am writing to express my strong enthusiasm for the AI Application Developer position. "
                "With over three years of specialized experience in Python, LangChain, and API development, "
                "I have architected modular LCEL pipelines and high-throughput FastAPI services that match your needs.\n\n"
                "At Innovatech Solutions, I spearheaded the development of retrieval-augmented generation pipelines "
                "using FAISS and FastAPI, improving query throughput and maintaining sub-second latency. "
                "While I look forward to further expanding my distributed container orchestration in cloud environments, "
                "my solid hands-on background in Docker and clean microservice architecture will allow me to deliver value from day one.\n\n"
                "I would welcome the opportunity to discuss how my technical expertise aligns with your mission.\n\n"
                "Sincerely,\n"
                "Alex Mercer"
            )

        if "proposal" in prompt_lower:
            return (
                "Hi there,\n\n"
                "I understand you are looking to build a clean, reliable multi-agent prototype using LangChain LCEL and FastAPI.\n\n"
                "My 3-step approach:\n"
                "1. Pipeline Setup: Build deterministic LCEL Runnable chains with robust JSON schema validation.\n"
                "2. Vector Indexing: Integrate FAISS retrieval with all-MiniLM embeddings for grounded context.\n"
                "3. API & UI: Expose modular endpoints and connect the interactive Streamlit interface.\n\n"
                "Timeline: Ready within 10 days.\n"
                "Price: $1,500 fixed.\n"
                "Proof point: Recently engineered JSBrother, an end-to-end multi-agent assistant with LangChain orchestration.\n\n"
                "Quick question: Do you prefer local FAISS vector indexing or a managed cloud vector database?\n\n"
                "Best regards,\n"
                "Alex Mercer"
            )

        # Default fallback
        if "hi" in prompt_lower or "hello" in prompt_lower:
            return "Hello! JSBrother LLM Client is online and ready."

        return f"Mock response for prompt: {prompt[:80]}"


# Module-level default instance
_default_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    """Get or create singleton LLMClient."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


def generate(
    prompt: str,
    system: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    client: Optional[LLMClient] = None,
) -> str:
    """
    Convenience function for LLM generation.
    Uses default configured client if not explicitly passed.
    """
    active_client = client or get_client()
    return active_client.generate(
        prompt=prompt,
        system=system,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
