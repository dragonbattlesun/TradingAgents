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
        ("us.anthropic.claude-opus-4-7-20251001-v1:0", True),  # versioned
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", False),
        # Regression guard — anchored end means trailing junk after the
        # numeric pair (without a `-` separator) cannot match. A model
        # named like 'claude-sonnet-4-6haiku' should not pass the gate.
        ("us.anthropic.claude-sonnet-4-6haiku", False),
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


@pytest.mark.unit
class TestBedrockFactoryWiring:
    def test_factory_returns_bedrock_client(self, monkeypatch):
        from tradingagents.llm_clients.factory import create_llm_client
        # Patch the constructor so get_llm() doesn't build a real boto3
        # client (no AWS_BEARER_TOKEN_BEDROCK in test env).
        monkeypatch.setattr(
            mod, "NormalizedChatBedrockConverse",
            lambda **kwargs: object(),
        )
        client = create_llm_client(
            "bedrock",
            "us.anthropic.claude-opus-4-7",
        )
        assert isinstance(client, mod.BedrockClient)
        assert client.model == "us.anthropic.claude-opus-4-7"
        # Exercise the factory→client→get_llm chain so the monkeypatch
        # actually does work — proves the wiring builds something
        # without hitting boto3.
        assert client.get_llm() is not None


@pytest.mark.unit
class TestApiKeyEnvMapping:
    def test_bedrock_maps_to_aws_bearer_token(self):
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        assert get_api_key_env("bedrock") == "AWS_BEARER_TOKEN_BEDROCK"
        # Case-insensitive — same convention used by other providers.
        assert get_api_key_env("BEDROCK") == "AWS_BEARER_TOKEN_BEDROCK"


@pytest.mark.unit
class TestBedrockModelCatalog:
    def test_quick_and_deep_options_have_three_entries_each(self):
        from tradingagents.llm_clients.model_catalog import get_model_options
        quick = get_model_options("bedrock", "quick")
        deep = get_model_options("bedrock", "deep")
        # Two concrete models + Custom model ID.
        assert len(quick) == 3
        assert len(deep) == 3
        # Custom escape hatch is always last.
        assert quick[-1] == ("Custom model ID", "custom")
        assert deep[-1] == ("Custom model ID", "custom")

    def test_concrete_model_ids_use_us_anthropic_prefix(self):
        from tradingagents.llm_clients.model_catalog import get_model_options
        for label, value in get_model_options("bedrock", "deep")[:-1]:
            assert value.startswith("us.anthropic.claude-")
        for label, value in get_model_options("bedrock", "quick")[:-1]:
            assert value.startswith("us.anthropic.claude-")


@pytest.mark.unit
class TestProviderKwargsForBedrock:
    """The trading graph's _get_provider_kwargs must pass anthropic_effort
    through for the bedrock provider so a single knob controls both."""

    def _make_graph_method(self):
        # Construct a stand-in object that has the relevant fields without
        # spinning up the full TradingAgentsGraph (which builds LLM clients
        # and a langgraph workflow).
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        return TradingAgentsGraph._get_provider_kwargs

    def test_anthropic_effort_flows_through_for_bedrock(self):
        method = self._make_graph_method()

        class Stub:
            def __init__(self):
                self.config = {"llm_provider": "bedrock", "anthropic_effort": "high"}

        kwargs = method(Stub())
        assert kwargs.get("effort") == "high"

    def test_no_effort_when_unset(self):
        method = self._make_graph_method()

        class Stub:
            def __init__(self):
                self.config = {"llm_provider": "bedrock"}

        kwargs = method(Stub())
        assert "effort" not in kwargs


@pytest.mark.unit
class TestCliProviderTable:
    def test_bedrock_row_present(self):
        from cli.utils import _llm_provider_table
        rows = _llm_provider_table()
        bedrock_rows = [row for row in rows if row[1] == "bedrock"]
        assert len(bedrock_rows) == 1
        display, key, base_url = bedrock_rows[0]
        assert display == "AWS Bedrock"
        assert key == "bedrock"
        # Bedrock derives its endpoint from region_name, not a URL.
        assert base_url is None
