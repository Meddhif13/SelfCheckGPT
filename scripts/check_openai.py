"""Small connectivity check for OpenAI API.

- Resolves API key from env or .secrets/openai.key
- Lists available models (best-effort)
- Runs a 1-line chat completion ("Say 'pong'.")

Usage:
  python scripts/check_openai.py [--model MODEL]

Exit codes:
  0 = success, API reachable and chat worked
  1 = failure
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time
import json
from typing import Optional

try:
    # New style SDK (>=1.0)
    from openai import OpenAI, APIError, RateLimitError
    import httpx
except Exception as e:  # pragma: no cover
    print("[check_openai] Failed to import dependencies. Ensure 'openai' and 'httpx' are installed.")
    raise


def resolve_api_key() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key.strip()
    # Try env file path
    key_file = os.getenv("OPENAI_API_KEY_FILE")
    if key_file and Path(key_file).exists():
        return Path(key_file).read_text(encoding="utf-8").strip()
    # Try repo-local secrets
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / ".secrets" / "openai.key",
        repo_root / ".env",
    ]
    for fp in candidates:
        if fp.exists():
            text = fp.read_text(encoding="utf-8")
            if "OPENAI_API_KEY=" in text:
                for line in text.splitlines():
                    if line.strip().startswith("OPENAI_API_KEY="):
                        value = line.split("=", 1)[1].strip().strip('"').strip("'")
                        if value:
                            return value
            else:
                value = text.strip()
                if value:
                    return value
    return None


def make_client(timeout: float = 20.0, max_retries: int = 2, *, no_proxy: bool = False) -> OpenAI:
    key = resolve_api_key()
    if not key:
        print("[check_openai] OPENAI_API_KEY not found. Set env var or create .secrets/openai.key")
        sys.exit(1)
    # Build an httpx client; optionally bypass proxies and HTTP/2
    http_client = httpx.Client(
        timeout=timeout,
        http2=False,
        transport=httpx.HTTPTransport(retries=max_retries),
        trust_env=not no_proxy,  # when False, ignores system env including proxies
    )
    return OpenAI(api_key=key, timeout=timeout, max_retries=max_retries, base_url="https://api.openai.com/v1", http_client=http_client)


def try_list_models(client: OpenAI) -> None:
    try:
        models = client.models.list()
        ids = [m.id for m in getattr(models, "data", [])][:5]
        print(f"[check_openai] models.list() OK. Showing up to 5: {ids}")
    except Exception as e:
        print(f"[check_openai] models.list() failed: {type(e).__name__}: {e}")


def try_small_chat(client: OpenAI, model: str) -> None:
    print(f"[check_openai] Trying small chat on model='{model}'...")
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'pong'."}],
            temperature=0.0,
        )
        text = res.choices[0].message.content or ""
        print("[check_openai] chat OK. Response:", json.dumps(text, ensure_ascii=False))
    except (RateLimitError, APIError) as e:
        print(f"[check_openai] chat API error: {type(e).__name__}: {e}")
        raise
    except Exception as e:
        print(f"[check_openai] chat failed: {type(e).__name__}: {e}")
        raise


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    args = parser.parse_args(argv)

    print("[check_openai] Python:", sys.version.replace("\n", " "))
    try:
        import openai as openai_pkg  # type: ignore
        print("[check_openai] openai SDK:", getattr(openai_pkg, "__version__", "unknown"))
    except Exception:
        pass

    # Show proxy envs if any
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if http_proxy or https_proxy:
        print(f"[check_openai] Proxy env detected HTTP_PROXY={bool(http_proxy)} HTTPS_PROXY={bool(https_proxy)}")
    else:
        print("[check_openai] No proxy env detected.")

    client = make_client()
    try_list_models(client)

    # quick retry loop for chat (handles transient 520/502)
    delays = [0.5, 1.0, 2.0]
    last_err: Optional[Exception] = None
    for i, d in enumerate([0.0] + delays):
        if d:
            time.sleep(d)
        try:
            try_small_chat(client, args.model)
            print("[check_openai] SUCCESS")
            return 0
        except Exception as e:
            last_err = e
            print(f"[check_openai] Attempt {i+1} failed; will retry..." if i < len(delays) else "[check_openai] No more retries.")
            continue

    # Try once more bypassing proxies and forcing HTTP/1.1
    print("[check_openai] Final attempt without proxies and HTTP/2...")
    try:
        client_np = make_client(no_proxy=True)
        try_small_chat(client_np, args.model)
        print("[check_openai] SUCCESS (no-proxy)")
        return 0
    except Exception as e:
        last_err = e

    if last_err:
        print(f"[check_openai] FAILED: {type(last_err).__name__}: {last_err}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
