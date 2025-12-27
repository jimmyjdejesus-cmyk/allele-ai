import os
from pathlib import Path
import re
import shutil
import tempfile

import pytest

from scripts.agent_safety_check import scan_file, find_trc_occurrences


def test_scan_file_detects_secrets(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)
    f = src_dir / "secrets.py"
    # write an assignment with a long token
    f.write_text("API_KEY = 'abcdefghijklmnopqrstuvwxyz012345'\n")

    # monkeypatch ROOT so scan_file reads relative paths normally
    monkeypatch.chdir(project_root)

    matches = scan_file(f)
    assert any('api[_-]?key' in m.lower() for m in matches)


def test_find_trc_occurrences(tmp_path, monkeypatch):
    project_root = tmp_path / "proj"
    src_dir = project_root / "src"
    src_dir.mkdir(parents=True)

    f1 = src_dir / "a.py"
    f1.write_text("trust_remote_code = True\n")

    f2 = src_dir / "b.json"
    f2.write_text('{"trust_remote_code": true}')

    # The finder currently looks at .py files only, so create a .py wrapper
    f2_py = src_dir / "b_py.py"
    f2_py.write_text('config = {"trust_remote_code": true}')

    monkeypatch.chdir(project_root)

    occurrences = find_trc_occurrences([src_dir])
    # paths may be returned relative to the provided base or absolute; just assert names
    names = {p.name if isinstance(p, Path) else Path(str(p)).name for p in occurrences}
    assert 'a.py' in names
    assert len(names) >= 1
