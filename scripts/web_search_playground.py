"""
Playground for testing the VertexAI adapter's Claude web search.

Assumes the adapter is already running on http://localhost:5001.
Copy-paste this whole file into a Jupyter cell and run it.

Only depends on `httpx` (already a project dependency): `pip install httpx`.
"""

import json

import httpx

# --- Config -----------------------------------------------------------------

BASE_URL = "http://localhost:5001"
# Latest Opus exposed by the adapter (there is no "4.8"; swap freely).
DEPLOYMENT = "claude-opus-4-6"
API_KEY = "dummy-key"  # any value works unless the adapter enforces auth
REGION = None  # e.g. "us-east5"; sent via x-upstream-extra-data when set
STREAM = True

PROMPT = "What are the top AI news today? Please cite your sources."

# Anthropic web search tool definition. `type` is required; optional fields
# include max_uses / allowed_domains / blocked_domains / user_location.
WEB_SEARCH_CONFIGURATION = {
    "type": "web_search_20250305",
    "max_uses": 5,
}

# --- Request ----------------------------------------------------------------

url = f"{BASE_URL}/openai/deployments/{DEPLOYMENT}/chat/completions"

headers = {"Api-Key": API_KEY, "Content-Type": "application/json"}
if REGION:
    headers["x-upstream-extra-data"] = json.dumps({"region": REGION})


def build_payload(stream: bool) -> dict:
    # `stream` is set per-call so the request always matches the reader below;
    # never rely on a shared global here, or a non-streaming reader may receive
    # an SSE stream (and vice versa).
    return {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 2048,
        "stream": stream,
        "tools": [
            {
                "type": "static_function",
                "static_function": {
                    "name": "web_search",
                    "configuration": WEB_SEARCH_CONFIGURATION,
                },
            }
        ],
    }


def _print_stages(stages: dict[int, dict]) -> None:
    for _, stage in sorted(stages.items()):
        name = stage.get("name") or "Stage"
        content = (stage.get("content") or "").strip()
        print(f"\n[{name}] {content}")


def _print_attachments(attachments: list[dict]) -> None:
    if not attachments:
        return
    print("\n\nSources:")
    for att in attachments:
        title = att.get("title") or "(untitled)"
        ref = att.get("url") or att.get("reference_url") or ""
        print(f"  - {title}: {ref}")


def run_streaming() -> None:
    stages: dict[int, dict] = {}
    attachments: list[dict] = []

    with httpx.stream(
        "POST", url, headers=headers, json=build_payload(True), timeout=120
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)

            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if content := delta.get("content"):
                    print(content, end="", flush=True)

                custom = delta.get("custom_content") or {}
                for stage in custom.get("stages", []):
                    entry = stages.setdefault(
                        stage.get("index", 0), {"name": None, "content": ""}
                    )
                    if stage.get("name"):
                        entry["name"] = stage["name"]
                    if stage.get("content"):
                        entry["content"] += stage["content"]
                attachments.extend(custom.get("attachments", []))

            if usage := chunk.get("usage"):
                print(f"\n\nUsage: {usage}")

    _print_stages(stages)
    _print_attachments(attachments)


def run_non_streaming() -> None:
    response = httpx.post(
        url, headers=headers, json=build_payload(False), timeout=120
    )
    response.raise_for_status()
    body = response.json()

    message = body["choices"][0]["message"]
    print(message.get("content") or "")

    custom = message.get("custom_content") or {}
    stages = {
        i: {"name": s.get("name"), "content": s.get("content")}
        for i, s in enumerate(custom.get("stages", []))
    }
    _print_stages(stages)
    _print_attachments(custom.get("attachments", []))

    if usage := body.get("usage"):
        print(f"\nUsage: {usage}")


if __name__ == "__main__":
    if STREAM:
        run_streaming()
    else:
        run_non_streaming()
