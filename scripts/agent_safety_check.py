#!/usr/bin/env python3
"""Basic agent safety check script (placeholder).

Scans repository for potential secret-like strings and flags files under AGENTS/ or src/agent modules.
"""
import os
import re
import sys
from pathlib import Path

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key"),
    re.compile(r"(?i)secret"),
    re.compile(r"(?i)aws[_-]?secret"),
    re.compile(r"(?i)password"),
    re.compile(r"(?i)private[_-]?key"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # JWT-like pattern: three base64url segments separated by dots (not exhaustive)
    re.compile(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
]

ROOT = Path(__file__).resolve().parents[1]

def scan_file(path: Path):
    text = path.read_text(errors='ignore')
    matches = []
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            # Heuristic: only flag if there is a likely secret value nearby
            # Look for quoted strings of length >= 16 or known prefixes
            # More precise heuristic: look for assignments of long quoted strings
            # Only consider long continuous token-like strings (no whitespace)
            assignment_pattern = re.compile(
                r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*['\"][A-Za-z0-9_\-\+/=]{16,}['\"]"
            )
            environ_default_pattern = re.compile(
                r"os\.environ\.get\(\s*['\"][A-Za-z0-9_]+['\"]\s*,\s*['\"][A-Za-z0-9_\-\+/=]{16,}['\"]\s*\)"
            )
            if assignment_pattern.search(text) or environ_default_pattern.search(text):
                matches.append(pattern.pattern)
    return matches


def find_trc_occurrences(bases):
    """Return list of files that contain an explicit trust_remote_code setting.

    Accept a list/iterable of base Path objects to search. Matches are
    tolerant of Python and JSON-style assignments, e.g.:
      trust_remote_code=True
      "trust_remote_code": true
      trust_remote_code = True
    """
    occurrences = []
    trc_pattern = re.compile(r"trust_remote_code\s*[:=]\s*(?:True|true)\b")
    for base in bases:
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            # skip this script file if ever included under src/ or AGENTS
            if p.resolve() == Path(__file__).resolve():
                continue
            try:
                text = p.read_text(errors='ignore')
            except Exception:
                continue
            if trc_pattern.search(text):
                try:
                    occurrences.append(p.relative_to(ROOT))
                except Exception:
                    # In tests or alternative roots, fall back to path relative to base
                    try:
                        occurrences.append(p.relative_to(base))
                    except Exception:
                        occurrences.append(p)
    return occurrences


def main():
    flagged = []
    # Focus scan on source code and agent specs to reduce false positives
    scan_paths = [ROOT / 'src', ROOT / 'AGENTS']
    for base in scan_paths:
        if not base.exists():
            continue
        for p in base.rglob("*.*"):
            if p.suffix in {'.py', '.env', '.cfg', '.yaml', '.yml', '.md'}:
                matches = scan_file(p)
                if matches:
                    flagged.append((p.relative_to(ROOT), matches))

    if flagged:
        print("Potential secret-like patterns found in files:")
        for f, m in flagged:
            print(f" - {f}: {', '.join(m)}")
        print("\nPlease ensure secrets are stored in GitHub secrets and not in the repository.")
        sys.exit(2)

    # Additional checks for known risky settings (non-fatal by default)
    trc_occurrences = find_trc_occurrences([ROOT / 'src', ROOT / 'AGENTS'])

    if trc_occurrences:
        print("Warning: explicit 'trust_remote_code=True' found in source files:")
        for p in trc_occurrences:
            print(f" - {p}")
        print("\nThis is a potentially dangerous configuration. If you want the check to fail CI on TRC detections, set AGENT_SAFETY_FAIL_ON_TRC=1 in the workflow environment.")
        # Fail only if explicitly requested via env var (gives teams control)
        if os.environ.get('AGENT_SAFETY_FAIL_ON_TRC') == '1':
            sys.exit(3)

    print("No obvious secret patterns found.")

if __name__ == '__main__':
    main()
