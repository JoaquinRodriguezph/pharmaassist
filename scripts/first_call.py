"""One real Gemini call, retried via app.llm.retry. Run from the project root:

    py scripts/first_call.py
"""

import logging
import os
import sys
from pathlib import Path

# Make `app` importable when running this file directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from google import genai

from app.llm.gemini_errors import classify_gemini_error
from app.llm.retry import call_with_retry


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL")

    if not api_key:
        print("Error: GEMINI_API_KEY is not set in the environment or .env file.", file=sys.stderr)
        sys.exit(1)
    if not model_name:
        print("Error: GEMINI_MODEL is not set in the environment or .env file.", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    prompt = "In two sentences, what does a pharmacy reorder level mean?"
    print(f"Sending prompt to model '{model_name}':\n\"{prompt}\"\n")

    try:
        response = call_with_retry(
            lambda: client.models.generate_content(model=model_name, contents=prompt),
            classify_gemini_error,
            max_attempts=4,
        )

        print("--- Response ---")
        print(response.text)
        print()

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            print("--- Token Usage ---")
            print(f"Prompt tokens: {getattr(meta, 'prompt_token_count', 'N/A')}")
            print(f"Response tokens: {getattr(meta, 'candidates_token_count', 'N/A')}")

            # Verify thoughts_token_count explicitly (preserving 0 if present)
            thoughts_count = getattr(meta, "thoughts_token_count", None)
            if thoughts_count is not None:
                print(f"Thoughts tokens: {thoughts_count}")
            else:
                print("Thoughts tokens: None")

            print(f"Total tokens: {getattr(meta, 'total_token_count', 'N/A')}")
        else:
            print("Token usage metadata not available in response.")

    except Exception as e:
        print(f"Fatal execution error: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()