import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from jsb.llm_client import get_client

client = get_client()
print(f"Client mode: {client.mode}, base_url: {client.base_url}")
try:
    resp = client.generate("Respond with ONLY the word PONG.", max_new_tokens=10)
    print("Response:", resp.strip())
except Exception as e:
    print("Error connecting to LLM:", e)
    sys.exit(1)
