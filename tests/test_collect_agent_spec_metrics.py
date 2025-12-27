import json
from unittest.mock import patch

import pytest

from scripts.collect_agent_spec_metrics import collect_metrics


class DummyResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json = json_data

    def json(self):
        return self._json


@patch('scripts.collect_agent_spec_metrics.requests.get')
@patch('scripts.collect_agent_spec_metrics.get_pr_changed_files')
def test_collect_metrics_basic(mock_changed, mock_get):
    # Mock PR list: one PR updated recently
    now = '2025-12-27T00:00:00Z'
    pr = {'number': 1, 'updated_at': now, 'labels': []}
    # First call returns PRs
    # get_recent_prs may call the pulls API twice (page 1, page 2) and pr_has_failure_comment will call comments
    mock_get.side_effect = [DummyResponse(200, [pr]), DummyResponse(200, []), DummyResponse(200, [{'body': 'some comment'}])]

    # Mock changed files to include an agent file
    mock_changed.return_value = ['src/phylogenic/my_agent.py']

    metrics = collect_metrics(days=7, token='token')
    assert metrics['inspected_prs'] == 1
    # No failure comment in the mocked comment -> flagged 0
    assert metrics['flagged_prs'] == 0


@patch('scripts.collect_agent_spec_metrics.requests.get')
@patch('scripts.collect_agent_spec_metrics.get_pr_changed_files')
def test_collect_metrics_flagged(mock_changed, mock_get):
    now = '2025-12-27T00:00:00Z'
    pr = {'number': 2, 'updated_at': now, 'labels': [{'name': 'allow-agent-without-spec'}]}
    # get_recent_prs may call pulls API twice (page 1, page 2), then comments endpoint
    mock_get.side_effect = [DummyResponse(200, [pr]), DummyResponse(200, []), DummyResponse(200, [{'body': '❌ Agent spec check failed: agent code changes require a matching agent spec attached under AGENTS/.'}])]

    mock_changed.return_value = ['AGENTS/special_agent.md']

    metrics = collect_metrics(days=7, token='token')
    assert metrics['inspected_prs'] == 1
    assert metrics['flagged_prs'] == 1
    assert metrics['override_count'] == 1
