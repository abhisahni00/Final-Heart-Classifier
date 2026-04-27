"""Ping Gemini from the terminal (API key only in your shell, not committed)."""
import os

from google import genai
from google.genai import errors as genai_errors

if __name__ == "__main__":
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            'Set the key in this terminal first, e.g.  $env:GEMINI_API_KEY="your-key"'
        )
    client = genai.Client(api_key=key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
    try:
        response = client.models.generate_content(
            model=model_name,
            contents="Hello! Are you working?",
        )
        print(response.text.strip() if response.text else response)
    except genai_errors.ClientError as e:
        msg = str(e).lower()
        if "429" in msg or "quota" in msg or "resource" in msg:
            print(
                "429 quota/rate limit. Try:\n"
                '  $env:GEMINI_MODEL="gemini-2.0-flash"\n'
                "https://ai.dev/rate-limit\n"
            )
        raise SystemExit(str(e)) from e
