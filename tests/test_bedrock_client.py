"""Tests for the AWS Bedrock LLM client."""

import pytest

from tradingagents.llm_clients import bedrock_client as mod
from tradingagents.llm_clients.validators import validate_model


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatBedrockConverse",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


@pytest.mark.unit
class TestBedrockValidator:
    def test_any_model_id_accepted_for_bedrock(self):
        """Bedrock model IDs vary by region/version suffix; treat like
        OpenRouter — accept any string."""
        assert validate_model("bedrock", "us.anthropic.claude-opus-4-7") is True
        assert validate_model("bedrock", "anthropic.claude-opus-4-7") is True
        assert validate_model("bedrock", "totally-made-up") is True


@pytest.mark.unit
class TestBedrockClient:
    def test_model_and_region_forwarded(self, monkeypatch):
        captured = _capture_kwargs(monkeypatch)
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        mod.BedrockClient(model="us.anthropic.claude-opus-4-7").get_llm()
        assert captured["kwargs"]["model"] == "us.anthropic.claude-opus-4-7"
        assert captured["kwargs"]["region_name"] == "us-east-1"

    def test_no_api_key_kwarg_leaked(self, monkeypatch):
        """Bedrock auth comes from boto3 (AWS_BEARER_TOKEN_BEDROCK),
        not a client kwarg — make sure we never forward api_key."""
        captured = _capture_kwargs(monkeypatch)
        mod.BedrockClient(
            model="us.anthropic.claude-opus-4-7",
            api_key="should-not-leak",
        ).get_llm()
        assert "api_key" not in captured["kwargs"]

    @pytest.mark.parametrize("model_id,expects_effort", [
        ("us.anthropic.claude-opus-4-7", True),
        ("us.anthropic.claude-sonnet-4-6", True),
        ("anthropic.claude-opus-4-7", True),  # no region prefix
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", False),
    ])
    def test_effort_gate_matches_anthropic_semantics(
        self, monkeypatch, model_id, expects_effort
    ):
        captured = _capture_kwargs(monkeypatch)
        mod.BedrockClient(model=model_id, effort="high").get_llm()
        if expects_effort:
            assert captured["kwargs"]["additional_model_request_fields"] == {
                "thinking": {"type": "high"}
            }
        else:
            assert "additional_model_request_fields" not in captured["kwargs"]
