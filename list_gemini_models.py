"""One-off: list models (from .streamlit/secrets.toml GEMINI_API_KEY). Run: .\\.venv\\Scripts\\python.exe list_gemini_models.py"""
import re
from pathlib import Path

from google import genai

def main() -> None:
    raw = Path(".streamlit/secrets.toml").read_text(encoding="utf-8")
    m = re.search(r'GEMINI_API_KEY\s*=\s*"([^"]+)"', raw)
    if not m or not m.group(1).strip():
        raise SystemExit("Set GEMINI_API_KEY in .streamlit/secrets.toml")
    key = m.group(1).strip()
    client = genai.Client(api_key=key)
    print("All models (short id = strip 'models/' prefix):")
    for model in client.models.list():
        name = getattr(model, "name", "") or ""
        short = name.removeprefix("models/")
        methods = getattr(model, "supported_generation_methods", None) or []
        flags = ",".join(methods) if methods else ""
        print(f"  {short}\t[{flags}]")


if __name__ == "__main__":
    main()
