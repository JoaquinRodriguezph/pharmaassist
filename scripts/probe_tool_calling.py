"""Show how each backend returns a tool call. Run before writing the tool loop.

    py scripts/probe_tool_calling.py            # Ollama only (free)
    py scripts/probe_tool_calling.py --gemini   # also ONE Gemini call (uses 1 of the daily free requests)

Prints package versions, the raw tool call each backend returns for one question, and, for
Ollama, the reply after a fake tool result is sent back.
"""

import argparse
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from app.llm.env import require_env
from app.tools.registry import TOOLS_BY_NAME

SPEC = TOOLS_BY_NAME["get_expiring_batches"]
QUESTION = "What batches are expiring in the next 30 days at Quezon City Hub? Quezon City Hub is branch_id 2."
FAKE_RESULT = {
    "rows": [{"product_name": "Test Syrup 60ml", "batch_number": "B999-1", "quantity": 5,
              "expiry_date": "2026-11-01", "days_left": 12}],
    "total_rows": 1,
    "truncated": False,
}


def version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


def probe_ollama() -> None:
    import ollama

    print(f"=== Ollama (package {version('ollama')})")
    client = ollama.Client(host=require_env("OLLAMA_HOST"))
    model = require_env("OLLAMA_MODEL")
    tool = {"type": "function", "function": SPEC.json_schema()}
    messages = [{"role": "user", "content": QUESTION}]
    first = client.chat(model=model, messages=messages, tools=[tool], options={"temperature": 0})
    print("message:", repr(first.message))
    calls = first.message.tool_calls or []
    print("tool_calls count:", len(calls))
    for call in calls:
        print("  name:", call.function.name, "| arguments type:", type(call.function.arguments).__name__,
              "| arguments:", call.function.arguments)
    print("prompt_eval_count:", first.prompt_eval_count, "| eval_count:", first.eval_count)
    if not calls:
        print("No tool call returned; skipping the second turn.")
        return
    for key in ("tool_name", "name"):
        follow_up = messages + [first.message, {"role": "tool", "content": json.dumps(FAKE_RESULT), key: SPEC.name}]
        try:
            second = client.chat(model=model, messages=follow_up, tools=[tool], options={"temperature": 0})
            print(f"second turn with {key!r}: content={second.message.content!r} | tool_calls={second.message.tool_calls}")
        except Exception as error:  # the probe reports, it does not decide
            print(f"second turn with {key!r}: ERROR {type(error).__name__}: {error}")


def probe_gemini(call_api: bool) -> None:
    print(f"\n=== Gemini (google-genai {version('google-genai')})")
    from google import genai
    from google.genai import types

    declaration = None
    for label, kwargs in (
        ("parameters_json_schema", {"parameters_json_schema": SPEC.parameters}),
        ("parameters", {"parameters": SPEC.parameters}),
    ):
        try:
            declaration = types.FunctionDeclaration(name=SPEC.name, description=SPEC.description, **kwargs)
            print(f"FunctionDeclaration({label}=...) builds: OK")
            break
        except Exception as error:
            print(f"FunctionDeclaration({label}=...) builds: ERROR {type(error).__name__}: {error}")
    if declaration is None or not call_api:
        if not call_api:
            print("API call skipped (pass --gemini to make one call).")
        return
    client = genai.Client(api_key=require_env("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(temperature=0, tools=[types.Tool(function_declarations=[declaration])])
    response = client.models.generate_content(model=require_env("GEMINI_MODEL"), contents=QUESTION, config=config)
    print("function_calls:", response.function_calls)
    print("candidate content:", repr(response.candidates[0].content))
    usage = response.usage_metadata
    print("usage:", getattr(usage, "prompt_token_count", None), getattr(usage, "candidates_token_count", None),
          getattr(usage, "thoughts_token_count", None))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gemini", action="store_true", help="make one real Gemini call")
    args = parser.parse_args()
    probe_ollama()
    probe_gemini(args.gemini)
    return 0


if __name__ == "__main__":
    sys.exit(main())