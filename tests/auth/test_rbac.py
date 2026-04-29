"""Tests for RBAC checker using SubjectAccessReview."""

from unittest.mock import MagicMock

import pytest

from rhoai_mcp.auth.rbac import RBACChecker, ToolPermission


@pytest.fixture
def mock_authz_api():
    return MagicMock()


@pytest.fixture
def checker(mock_authz_api):
    return RBACChecker(mock_authz_api)


class TestToolPermission:
    def test_create(self):
        perm = ToolPermission(api_group="", resource="namespaces", verb="list")
        assert perm.api_group == ""
        assert perm.resource == "namespaces"
        assert perm.verb == "list"

    def test_from_dict(self):
        perm = ToolPermission.from_dict(
            {"apiGroup": "kubeflow.org", "resource": "notebooks", "verb": "get"}
        )
        assert perm.api_group == "kubeflow.org"

    def test_from_dict_missing_keys_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            ToolPermission.from_dict({"apiGroup": "", "resource": "pods"})

    def test_cache_key(self):
        p1 = ToolPermission(api_group="", resource="namespaces", verb="list")
        p2 = ToolPermission(api_group="", resource="namespaces", verb="list")
        assert p1.cache_key == p2.cache_key

        p3 = ToolPermission(api_group="", resource="namespaces", verb="create")
        assert p1.cache_key != p3.cache_key


class TestRBACChecker:
    def test_check_single_permission_allowed(self, checker, mock_authz_api):
        review_result = MagicMock()
        review_result.status.allowed = True
        mock_authz_api.create_subject_access_review.return_value = review_result

        perm = ToolPermission(api_group="", resource="namespaces", verb="list")
        result = checker.check_permission("alice", ["team-a"], perm)
        assert result is True

    def test_check_single_permission_denied(self, checker, mock_authz_api):
        review_result = MagicMock()
        review_result.status.allowed = False
        mock_authz_api.create_subject_access_review.return_value = review_result

        perm = ToolPermission(api_group="", resource="namespaces", verb="delete")
        result = checker.check_permission("alice", ["team-a"], perm)
        assert result is False

    def test_filter_tools_all_allowed(self, checker, mock_authz_api):
        review_result = MagicMock()
        review_result.status.allowed = True
        mock_authz_api.create_subject_access_review.return_value = review_result

        tool_perms = {
            "list_projects": [ToolPermission("", "namespaces", "list")],
            "create_project": [ToolPermission("", "namespaces", "create")],
        }
        allowed = checker.filter_tools("alice", ["team-a"], tool_perms)
        assert allowed == {"list_projects", "create_project"}

    def test_filter_tools_partial(self, checker, mock_authz_api):
        def sar_side_effect(body):
            result = MagicMock()
            verb = body.spec.resource_attributes.verb
            result.status.allowed = verb == "list"
            return result

        mock_authz_api.create_subject_access_review.side_effect = sar_side_effect

        tool_perms = {
            "list_projects": [ToolPermission("", "namespaces", "list")],
            "delete_project": [ToolPermission("", "namespaces", "delete")],
        }
        allowed = checker.filter_tools("alice", ["team-a"], tool_perms)
        assert allowed == {"list_projects"}

    def test_filter_tools_multi_perm_requires_all(self, checker, mock_authz_api):
        def sar_side_effect(body):
            result = MagicMock()
            verb = body.spec.resource_attributes.verb
            result.status.allowed = verb == "list"
            return result

        mock_authz_api.create_subject_access_review.side_effect = sar_side_effect

        tool_perms = {
            "complex_tool": [
                ToolPermission("", "namespaces", "list"),
                ToolPermission("", "namespaces", "create"),
            ],
        }
        allowed = checker.filter_tools("alice", ["team-a"], tool_perms)
        assert allowed == set()  # create is denied, so tool is hidden

    def test_deduplicates_sar_calls(self, checker, mock_authz_api):
        review_result = MagicMock()
        review_result.status.allowed = True
        mock_authz_api.create_subject_access_review.return_value = review_result

        tool_perms = {
            "tool_a": [ToolPermission("", "namespaces", "list")],
            "tool_b": [ToolPermission("", "namespaces", "list")],
        }
        checker.filter_tools("alice", [], tool_perms)
        # Same permission used by two tools -- only one SAR call
        assert mock_authz_api.create_subject_access_review.call_count == 1
