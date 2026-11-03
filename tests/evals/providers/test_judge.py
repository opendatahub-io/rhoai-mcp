"""Tests for GoogleJudgeLLM client initialization branches."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestEnMaaSJudge:
    @patch("evals.providers.judge.OpenAIJudgeLLM")
    def test_default_judge_uses_deepseek_on_enmaas(
        self, mock_judge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evals.config import EvalConfig
        from evals.providers import create_judge_llm

        monkeypatch.setenv("ENMAAS_KEY", "gateway-key")
        config = EvalConfig(_env_file=None, _env_prefix="RHOAI_TEST_")
        create_judge_llm(config)
        mock_judge.assert_called_once_with(
            model_name="rits/deepseek-ai/DeepSeek-V4.1-Flash-1017",
            api_key="gateway-key",
            base_url="https://api.enmaas.devshift.net/v1",
        )

    @patch("evals.providers.judge.OpenAIJudgeLLM")
    def test_enmaas_uses_gateway_key_and_endpoint(
        self, mock_judge: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evals.config import EvalConfig
        from evals.providers import create_judge_llm

        monkeypatch.setenv("ENMAAS_KEY", "gateway-key")
        config = EvalConfig(_env_file=None, eval_provider="enmaas", eval_model="gpt-5.6-sol")
        create_judge_llm(config)
        mock_judge.assert_called_once_with(
            model_name="gpt-5.6-sol",
            api_key="gateway-key",
            base_url="https://api.enmaas.devshift.net/v1",
        )

    def test_enmaas_missing_key_fails_without_provider_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from evals.config import EvalConfig
        from evals.providers import create_judge_llm

        monkeypatch.delenv("ENMAAS_KEY", raising=False)
        config = EvalConfig(_env_file=None, eval_provider="enmaas", eval_api_key="")
        with pytest.raises(ValueError, match="ENMAAS_KEY"):
            create_judge_llm(config)


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
    def test_vertex_no_key_no_project_raises(self, _mock_client_cls: MagicMock) -> None:
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
