"""
Tests for JSBrother LLM Client
Verifies mock mode, canned outputs, remote mode with simulated endpoint,
Bearer authentication, timeout, and retry handling.
"""

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import pytest

from jsb.llm_client import LLMClient, generate


# -----------------------------------------------------------------------------
# Simulated Remote Server for testing remote mode
# -----------------------------------------------------------------------------
class MockRemoteHandler(BaseHTTPRequestHandler):
    fail_first_n_requests = 0
    request_count = 0

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "model": "Qwen/Qwen2.5-7B-Instruct"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        MockRemoteHandler.request_count += 1

        # Check Authorization header
        auth_header = self.headers.get("Authorization", "")
        if auth_header != "Bearer test-secret-key":
            self.send_response(403)
            self.end_headers()
            self.wfile.write(b'{"detail": "Invalid API Key"}')
            return

        # Check path
        if self.path != "/generate":
            self.send_response(404)
            self.end_headers()
            return

        # Simulate transient failure if configured
        if MockRemoteHandler.request_count <= MockRemoteHandler.fail_first_n_requests:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b'{"detail": "Temporary GPU Out of Memory"}')
            return

        content_len = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_len)
        body = json.loads(post_data.decode("utf-8"))

        prompt = body.get("prompt", "")
        if "hi" in prompt.lower():
            reply_text = "Remote Qwen says: Hello from GPU!"
        else:
            reply_text = f"Remote output for: {prompt[:30]}"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"text": reply_text}).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence default stderr logging during tests
        pass


@pytest.fixture(scope="module")
def mock_remote_server():
    server = HTTPServer(("127.0.0.1", 0), MockRemoteHandler)
    port = server.server_port
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()


# -----------------------------------------------------------------------------
# Unit Tests
# -----------------------------------------------------------------------------
def test_mock_generate_hi():
    """Verify generate('hi') works in mock mode."""
    client = LLMClient(mode="mock")
    response = client.generate("hi")
    assert "JSBrother LLM Client is online" in response
    assert isinstance(response, str)


def test_mock_convenience_function():
    """Verify top-level generate() works in mock mode."""
    client = LLMClient(mode="mock")
    response = generate("hello there", client=client)
    assert "JSBrother LLM Client is online" in response


def test_mock_health_check():
    """Verify mock health check returns status ok."""
    client = LLMClient(mode="mock")
    health = client.check_health()
    assert health["status"] == "ok"
    assert health["mode"] == "mock"


def test_mock_profile_analyzer_output():
    """Verify mock generates valid ProfileJSON for CV extraction."""
    client = LLMClient(mode="mock")
    raw = client.generate("Please extract CV text to ProfileJSON format.")
    data = json.loads(raw)
    assert "name" in data
    assert "skills" in data
    assert "python" in [s.lower() for s in data["skills"]]
    assert "years_experience" in data
    assert data["years_experience"] > 0


def test_mock_posting_analyzer_output():
    """Verify mock generates valid PostingJSON for job and project modes."""
    client = LLMClient(mode="mock")
    # Job mode
    job_raw = client.generate("Extract job posting to PostingJSON.")
    job_data = json.loads(job_raw)
    assert job_data["type"] == "job"
    assert "must_have" in job_data
    assert len(job_data["must_have"]) > 0

    # Project mode
    fl_raw = client.generate("Extract freelance project posting to PostingJSON.")
    fl_data = json.loads(fl_raw)
    assert fl_data["type"] == "project"
    assert fl_data["budget"] is not None


def test_mock_matcher_output():
    """Verify mock generates valid Matcher reasons and advice."""
    client = LLMClient(mode="mock")
    raw = client.generate("Matcher: phrase reasons and advice for score.")
    data = json.loads(raw)
    assert "reasons" in data
    assert len(data["reasons"]) == 3
    assert "advice" in data


def test_mock_writer_output():
    """Verify mock generates cover letter and proposal drafts."""
    client = LLMClient(mode="mock")
    cl = client.generate("Writer: craft a tailored cover letter.")
    assert "Dear" in cl
    assert "Sincerely" in cl or "Best" in cl or "Alex Mercer" in cl

    prop = client.generate("Writer: draft a freelance proposal.")
    assert "approach" in prop.lower()
    assert "timeline" in prop.lower()


def test_remote_generate_hi(mock_remote_server):
    """Verify generate('hi') works in remote mode against FastAPI server."""
    MockRemoteHandler.request_count = 0
    MockRemoteHandler.fail_first_n_requests = 0

    client = LLMClient(
        mode="remote",
        base_url=mock_remote_server,
        api_key="test-secret-key",
        timeout=5,
        max_retries=2,
    )

    # 1. Check health endpoint
    health = client.check_health()
    assert health["status"] == "ok"
    assert "Qwen" in health["model"]

    # 2. Test generate("hi")
    response = client.generate("hi")
    assert "Remote Qwen says: Hello from GPU!" in response


def test_remote_retry_on_server_error(mock_remote_server):
    """Verify client retries automatically when remote server fails intermittently."""
    MockRemoteHandler.request_count = 0
    # Simulate 1 transient failure, then success on second attempt
    MockRemoteHandler.fail_first_n_requests = 1

    client = LLMClient(
        mode="remote",
        base_url=mock_remote_server,
        api_key="test-secret-key",
        timeout=5,
        max_retries=2,
    )

    response = client.generate("hi")
    assert "Remote Qwen says: Hello from GPU!" in response
    assert MockRemoteHandler.request_count == 2


def test_remote_invalid_api_key(mock_remote_server):
    """Verify remote call fails cleanly when wrong API key is supplied."""
    MockRemoteHandler.request_count = 0
    MockRemoteHandler.fail_first_n_requests = 0

    client = LLMClient(
        mode="remote",
        base_url=mock_remote_server,
        api_key="wrong-key",
        timeout=2,
        max_retries=0,
    )

    with pytest.raises(RuntimeError) as exc_info:
        client.generate("hi")
    assert "403" in str(exc_info.value)
