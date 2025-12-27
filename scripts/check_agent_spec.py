#!/usr/bin/env python3
"""Check that PRs which modify agent code include an agent spec.

Usage:
  python scripts/check_agent_spec.py --pr 123
  python scripts/check_agent_spec.py --local

Exit codes:
 - 0 OK (specs present or no agent files changed)
 - 2 API / invocation error
 - 4 Missing spec for agent code change

This script is intentionally lightweight and has a `--local` mode to let
developers run the check before pushing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import requests

REPO_OWNER = os.environ.get("GITHUB_REPOSITORY", "").split("/")[0]
REPO_NAME = os.environ.get("GITHUB_REPOSITORY", "").split("/")[1] if "/" in os.environ.get("GITHUB_REPOSITORY", "") else "Phylogenic-AI-Agents"

AGENT_FILE_PATTERNS = [re.compile(r"^AGENTS/", re.I), re.compile(r"src/.*/agent.*\.py$", re.I), re.compile(r"agent", re.I)]


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{p.stderr}")
    return p.stdout.strip()


def detect_agent_changes(changed_files: Iterable[str]) -> List[str]:
    agents = []
    for f in changed_files:
        for pat in AGENT_FILE_PATTERNS:
            if pat.search(f):
                agents.append(f)
                break
    return agents


def find_matching_spec_for_agent(agent_path: str) -> Optional[Path]:
    # Heuristic: look for files under AGENTS/ that contain the agent base name or a spec file
    base = Path(agent_path).stem
    root = Path.cwd()
    for p in root.rglob('AGENTS/**'):
        # skip if not a file
        if not p.is_file():
            continue
        name = p.name.lower()
        try:
            text = p.read_text(errors='ignore')
        except Exception:
            continue
        if base.lower() in name or base.lower() in text.lower() or 'agent name' in text.lower():
            return p
    return None


def get_pr_changed_files(pr: int, token: str) -> List[str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/pulls/{pr}/files"
    files = []
    page = 1
    while True:
        r = requests.get(url, params={"page": page, "per_page": 100}, headers=headers)
        if r.status_code != 200:
            raise RuntimeError(f"GitHub API error: {r.status_code} {r.text}")
        data = r.json()
        # If the response is empty or does not look like a list of files
        # (i.e., lacks 'filename'), stop paginating and treat as no more files.
        if not data or not any(isinstance(entry, dict) and 'filename' in entry for entry in data):
            break
        for entry in data:
            files.append(entry.get('filename'))
        # GitHub paginates; if we got a partial page then we can stop
        if len(data) < 100:
            break
        page += 1
    return files


def get_pr_labels(pr: int, token: str) -> List[str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    url = f"https://api.github.com/repos/{REPO_OWNER}/{REPO_NAME}/issues/{pr}/labels"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error: {r.status_code} {r.text}")
    data = r.json()
    return [d.get('name') for d in data]


def get_changed_files_local(base_branch: str = 'dev') -> List[str]:
    # get list of changed files relative to base_branch
    try:
        run_cmd(['git', 'fetch', 'origin', base_branch])
    except RuntimeError:
        # ignore fetch errors; fall back to current diff
        pass
    out = run_cmd(['git', 'diff', '--name-only', f'origin/{base_branch}...HEAD'])
    if not out:
        return []
    return out.splitlines()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr', type=int, help='Pull request number (uses GitHub API)')
    parser.add_argument('--local', action='store_true', help='Run in local mode (git diff against origin/dev)')
    parser.add_argument('--base', default='dev', help='Base branch to diff against for --local')
    args = parser.parse_args(argv)

    token = os.environ.get('GITHUB_TOKEN', '')
    try:
        if args.local:
            changed = get_changed_files_local(args.base)
        elif args.pr:
            changed = get_pr_changed_files(args.pr, token)
        else:
            # try to infer from event
            event_path = os.environ.get('GITHUB_EVENT_PATH')
            if event_path and Path(event_path).exists():
                payload = json.loads(Path(event_path).read_text())
                # attempt to extract changed files from the event (not always present)
                pr = payload.get('number') or payload.get('pull_request', {}).get('number')
                if pr:
                    changed = get_pr_changed_files(pr, token)
                else:
                    print('No PR number found in event; running local mode.')
                    changed = get_changed_files_local(args.base)
            else:
                print('No PR specified and no GITHUB_EVENT_PATH; run with --local or --pr')
                return 2

        agent_changes = detect_agent_changes(changed)
        if not agent_changes:
            print('No agent-related changes detected.')
            return 0

        # For each agent change, confirm a spec exists
        missing = []
        for af in agent_changes:
            spec = find_matching_spec_for_agent(af)
            if spec:
                print(f'Found spec for {af}: {spec}')
            else:
                missing.append(af)

        if not missing:
            print('OK — agent spec(s) found for changed agent code')
            return 0

        # Check for overrides
        override_label = 'allow-agent-without-spec'
        labels = []
        if args.pr:
            try:
                labels = get_pr_labels(args.pr, token)
            except Exception as e:
                print(f'Warning: failed to fetch PR labels: {e}')
        env_override = os.environ.get('AGENT_SAFETY_IGNORE_SPEC') == '1'
        if env_override:
            print('WARNING: Missing specs but AGENT_SAFETY_IGNORE_SPEC=1 set; continuing (override)')
            return 0
        if override_label in labels:
            print(f'WARNING: Missing specs but PR has label {override_label}; continuing (override)')
            return 0

        print('\nERROR: Agent code changes detected without matching agent spec(s):')
        for m in missing:
            print(f' - {m}')
        print('\nAttach a filled agent spec under AGENTS/ (see AGENTS/TEMPLATES/agent_spec_template.md)')
        print(f'Or add label "{override_label}" to the PR to temporarily bypass (requires reviewer permission).')
        return 4

    except Exception as e:
        print(f'Fatal error while checking agent specs: {e}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
