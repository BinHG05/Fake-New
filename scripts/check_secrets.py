"""Small dependency-free secret guard for source-controlled text files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {".git", ".dvc", ".pytest_cache", "data", "models", "artifacts", "labels_storage", "archive"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "google_service_account": re.compile(r'"private_key"\s*:\s*"-----BEGIN'),
    "hardcoded_api_key": re.compile(r"(?:api[_-]?key|secret|token)\s*[=:]\s*['\"][A-Za-z0-9_\-]{20,}", re.I),
}


def main() -> None:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED for part in path.parts):
            continue
        if path.name in {".env", "service-account.json"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{name}: {path.relative_to(ROOT)}")
    if findings:
        raise SystemExit("Potential secrets found:\n" + "\n".join(findings))
    print("Secret scan passed.")


if __name__ == "__main__":
    main()
