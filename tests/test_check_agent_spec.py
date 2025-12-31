import os
import json
import pytest
from unittest.mock import patch
from pathlib import Path

from scripts import check_agent_spec as cas


class DummyResponse:
    def __init__(self, status, json_data):
        self.status_code = status
        self._json = json_data

    def json(self):
        return self._json


@patch('scripts.check_agent_spec.requests.get')
def test_pr_no_agent_changes(mock_get, tmp_path, monkeypatch):
    # PR files: no agent files
    mock_get.return_value = DummyResponse(200, [])
    files = cas.get_pr_changed_files(1, token='')
    assert files == []


@patch('scripts.check_agent_spec.requests.get')
def test_pr_agent_change_without_spec(mock_get, tmp_path, monkeypatch):
    # PR files: agent file changed
    mock_get.side_effect = [DummyResponse(200, [{'filename': 'src/phylogenic/my_agent.py'}]), DummyResponse(200, [])]
    # ensure no AGENTS spec exists
    with patch('scripts.check_agent_spec.find_matching_spec_for_agent') as mock_find:
        mock_find.return_value = None
        code = cas.main(['--pr', '1'])
        assert code == 4


@patch('scripts.check_agent_spec.requests.get')
def test_pr_agent_change_with_spec(mock_get, tmp_path, monkeypatch):
    mock_get.side_effect = [DummyResponse(200, [{'filename': 'src/phylogenic/my_agent.py'}]), DummyResponse(200, [])]
    with patch('scripts.check_agent_spec.find_matching_spec_for_agent') as mock_find:
        mock_find.return_value = Path('AGENTS/my_agent_spec.md')
        code = cas.main(['--pr', '1'])
        assert code == 0


@patch('scripts.check_agent_spec.requests.get')
def test_pr_agent_change_with_label_override(mock_get, tmp_path, monkeypatch):
    mock_get.side_effect = [DummyResponse(200, [{'filename': 'src/phylogenic/my_agent.py'}]), DummyResponse(200, [{'name': 'allow-agent-without-spec'}])]
    with patch('scripts.check_agent_spec.find_matching_spec_for_agent') as mock_find:
        mock_find.return_value = None
        code = cas.main(['--pr', '1'])
        assert code == 0


def test_local_mode_no_agent(tmp_path, monkeypatch, capsys):
    # Create empty git repo locally to simulate no changes
    monkeypatch.chdir(tmp_path)
    Path('.git').mkdir()
    # Make local mode use no changed files by mocking get_changed_files_local
    with patch('scripts.check_agent_spec.get_changed_files_local') as mock_local:
        mock_local.return_value = []
        code = cas.main(['--local'])
        assert code == 0


@patch('scripts.check_agent_spec.requests.get')
def test_pr_non_code_agents_changed_ignored(mock_get, tmp_path, monkeypatch):
    # PR contains AGENTS docs and templates and workflow files (non-code). These should not
    # trigger the agent-spec requirement because they are docs/templates.
    mock_get.side_effect = [
        DummyResponse(200, [
            {"filename": ".github/ISSUE_TEMPLATES/agent_incident.md"},
            {"filename": "AGENTS.md"},
            {"filename": "AGENTS/TEMPLATES/agent_spec_template.md"},
            {"filename": "docs/agents.md"},
            {"filename": "scripts/agent_safety_check.py"},
        ]),
    ]

    # Mock find_matching_spec to ensure it's not called for docs
    with patch('scripts.check_agent_spec.find_matching_spec_for_agent') as mock_find:
        mock_find.return_value = None
        code = cas.main(['--pr', '1'])
        # scripts/agent_safety_check.py contains 'agent' in name but is a utility; we do not
        # require a spec for helper scripts — expecting no agent-spec enforcement here.
        assert code == 0
