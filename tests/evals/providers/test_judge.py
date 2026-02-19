"""Tests for GoogleJudgeLLM client initialization branches."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGoogleJudgeLLMInit:
    """Tests for GoogleJudgeLLM.__init__ use_vertex branches."""

    @patch("google.genai.Client")
    def test_vertex_express_mode_with_api_key(self, mock_client_cls: MagicMock) -> None:
        """use_vertex=True + api_key → Vertex AI Express mode."""
        from evals.providers.judge import GoogleJudgeLLM

        GoogleJudgeLLM(
            model_name="gemini-2.5-flash",
            api_key="test-key",
            use_vertex=True,
        )

        mock_client_cls.assert_called_once_with(vertexai=True, api_key="test-key")

    @patch("google.genai.Client")
    def test_vertex_adc_with_project(self, mock_client_cls: MagicMock) -> None:
        """use_vertex=True + vertex_project_id (no api_key) → ADC mode."""
        from evals.providers.judge import GoogleJudgeLLM

        GoogleJudgeLLM(
            model_name="gemini-2.5-flash",
            vertex_project_id="my-project",
            vertex_location="us-east1",
            use_vertex=True,
        )

        mock_client_cls.assert_called_once_with(
            vertexai=True,
            project="my-project",
            location="us-east1",
        )

    @patch("google.genai.Client")
    def test_vertex_no_key_no_project_raises(self, mock_client_cls: MagicMock) -> None:
        """use_vertex=True with neither api_key nor vertex_project_id → ValueError."""
        from evals.providers.judge import GoogleJudgeLLM

        with pytest.raises(ValueError, match="requires an API key or vertex_project_id"):
            GoogleJudgeLLM(
                model_name="gemini-2.5-flash",
                use_vertex=True,
            )

    @patch("google.genai.Client")
    def test_non_vertex_uses_api_key(self, mock_client_cls: MagicMock) -> None:
        """use_vertex=False → standard genai.Client with api_key."""
        from evals.providers.judge import GoogleJudgeLLM

        GoogleJudgeLLM(
            model_name="gemini-2.5-flash",
            api_key="test-key",
        )

        mock_client_cls.assert_called_once_with(api_key="test-key")

    @patch("google.genai.Client")
    def test_vertex_express_ignores_project_id(self, mock_client_cls: MagicMock) -> None:
        """use_vertex=True + api_key + vertex_project_id → Express mode (api_key wins)."""
        from evals.providers.judge import GoogleJudgeLLM

        GoogleJudgeLLM(
            model_name="gemini-2.5-flash",
            api_key="test-key",
            vertex_project_id="my-project",
            use_vertex=True,
        )

        mock_client_cls.assert_called_once_with(vertexai=True, api_key="test-key")
