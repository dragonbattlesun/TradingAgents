# AWS Bedrock LLM Provider — Design

**Date:** 2026-06-03
**Status:** Approved (pending implementation)
**Owner:** dragonbattesun

## Goal

Add a new `bedrock` LLM provider to TradingAgents so the framework can drive
Anthropic Claude models hosted on AWS Bedrock, reusing the AWS Bedrock bearer
token already exported in the user's shell (`AWS_BEARER_TOKEN_BEDROCK` +
`AWS_REGION=us-east-1`). No new credentials are required for the primary user.

## Non-goals

- No support for Bedrock-hosted non-Anthropic families (Llama, Mistral,
  Cohere, etc.). They can be reached via "Custom model ID" but we do not
  validate or showcase them.
- No SigV4 / IAM-key flow. `AWS_BEARER_TOKEN_BEDROCK` is the only auth path.
- No README changes. Internal use only.
- No integration tests against the live Bedrock endpoint.

## Context

The user's shell already has, via `~/.zshrc` and `~/.claude/bedrock.env`:

| Variable                    | Value                                       |
| --------------------------- | ------------------------------------------- |
| `AWS_BEARER_TOKEN_BEDROCK`  | (token, sourced from `~/.claude/bedrock.env`) |
| `AWS_REGION`                | `us-east-1`                                 |
| `ANTHROPIC_AUTH_TOKEN`      | explicitly `unset`                          |
| `ANTHROPIC_BASE_URL`        | explicitly `unset`                          |

There is no `ANTHROPIC_API_KEY` (direct Anthropic API), so the existing
`anthropic` provider in TradingAgents cannot run. The repo has `langchain-*`
packages for OpenAI / Anthropic / Google but **no `langchain-aws`** yet.

## Architecture

A new `bedrock` provider sits parallel to `anthropic` in the existing client
factory. Both share `BaseLLMClient`; nothing else.

```
.env: TRADINGAGENTS_LLM_PROVIDER=bedrock
       TRADINGAGENTS_DEEP_THINK_LLM=us.anthropic.claude-opus-4-7
       TRADINGAGENTS_QUICK_THINK_LLM=us.anthropic.claude-haiku-4-5-20251001-v1:0
shell: AWS_BEARER_TOKEN_BEDROCK=...   (reused from Claude Code setup)
       AWS_REGION=us-east-1

TradingAgentsGraph
  → factory.create_llm_client("bedrock", model)
  → BedrockClient.get_llm()
  → NormalizedChatBedrockConverse(model=..., region_name=...)
  → langchain_aws.ChatBedrockConverse
  → boto3 bedrock-runtime client (auto-reads AWS_BEARER_TOKEN_BEDROCK + AWS_REGION)
```

## Files touched

| File                                              | Change kind | Summary                                           |
| ------------------------------------------------- | ----------- | ------------------------------------------------- |
| `pyproject.toml`                                  | edit        | Add `langchain-aws>=0.2.0` to `dependencies`      |
| `tradingagents/llm_clients/bedrock_client.py`     | new         | `BedrockClient` + `NormalizedChatBedrockConverse` |
| `tradingagents/llm_clients/factory.py`            | edit        | Add `if provider_lower == "bedrock"` branch      |
| `tradingagents/llm_clients/api_key_env.py`        | edit        | Map `"bedrock" -> "AWS_BEARER_TOKEN_BEDROCK"`    |
| `tradingagents/llm_clients/model_catalog.py`      | edit        | `MODEL_OPTIONS["bedrock"]` (3 model IDs + Custom) |
| `tradingagents/llm_clients/validators.py`         | edit        | Add `"bedrock"` to skip-validation tuple         |
| `tradingagents/graph/trading_graph.py`            | edit        | Mirror `anthropic_effort` read in `bedrock` branch of `_get_provider_kwargs` |
| `cli/utils.py`                                    | edit        | Add `("AWS Bedrock", "bedrock", None)` row       |
| `.env.example`                                    | edit        | Add commented Bedrock example block              |
| `tests/test_bedrock_client.py`                    | new         | 3 unit tests                                     |

`default_config.py` is intentionally **not** changed — Bedrock is opt-in via
env vars, defaults stay on OpenAI / GPT-5.5.

## BedrockClient implementation

`tradingagents/llm_clients/bedrock_client.py`:

```python
import os
import re
from typing import Any

from langchain_aws import ChatBedrockConverse

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_tokens", "temperature", "callbacks",
    # api_key / http_client intentionally NOT forwarded — auth comes from
    # boto3 (AWS_BEARER_TOKEN_BEDROCK), not a client kwarg.
)

# Bedrock model IDs may carry a region prefix: 'us.', 'apac.', 'eu.'
_EFFORT_PATTERN = re.compile(r"^(?:[a-z]{2,3}\.)?anthropic\.claude-(opus|sonnet)-\d+-\d+")


def _supports_effort(model_id: str) -> bool:
    """Whether the Bedrock-hosted Claude model accepts the effort param.

    Mirrors the direct-Anthropic gate but matches Bedrock model IDs. Haiku
    does not support effort on either path; new Opus/Sonnet versions
    inherit support automatically via the regex.
    """
    return bool(_EFFORT_PATTERN.match(model_id.lower()))


class NormalizedChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock via the Converse API.

    Auth: boto3's standard credential chain. The convention this project
    uses is AWS_BEARER_TOKEN_BEDROCK + AWS_REGION (already exported in
    users' shells for Claude Code).
    """

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        llm_kwargs: dict = {"model": self.model}
        if region:
            llm_kwargs["region_name"] = region

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Anthropic's "effort" maps to Converse's additional_model_request_fields.
        effort = self.kwargs.get("effort")
        if effort and _supports_effort(self.model):
            llm_kwargs["additional_model_request_fields"] = {
                "thinking": {"type": effort},
            }

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("bedrock", self.model)
```

### Decisions baked in

- **No `api_key` kwarg.** Bedrock auth flows through boto3, not a client
  parameter. Forwarding `api_key` here would be dead weight at best and
  confusing at worst.
- **Region resolution order:** `AWS_REGION` → `AWS_DEFAULT_REGION` → fall
  through to boto3's own resolution (`~/.aws/config`).
- **Effort transformation:** direct Anthropic uses a top-level `effort`
  parameter; Bedrock Converse accepts `thinking={"type": effort}` inside
  `additional_model_request_fields`. The mapping is one-line and isolated.
- **Effort model gate:** matches direct Anthropic semantics (Haiku skipped).
  Regex tolerates an optional region prefix.
- **`base_url` is accepted but ignored.** Bedrock's endpoint is region-derived,
  not a URL the user sets per call.

## Factory branch

`tradingagents/llm_clients/factory.py`, after the `anthropic` branch:

```python
    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return BedrockClient(model, base_url, **kwargs)
```

## Model catalog

`tradingagents/llm_clients/model_catalog.py`:

```python
_BEDROCK_MODELS: Dict[str, List[ModelOption]] = {
    "quick": [
        ("Claude Haiku 4.5 (Bedrock) - Fastest, near-frontier",
         "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("Claude Sonnet 4.6 (Bedrock) - Speed/intelligence balance",
         "us.anthropic.claude-sonnet-4-6"),
        ("Custom model ID", "custom"),
    ],
    "deep": [
        ("Claude Opus 4.7 (Bedrock) - Frontier, long-running agents",
         "us.anthropic.claude-opus-4-7"),
        ("Claude Sonnet 4.6 (Bedrock) - Speed/intelligence balance",
         "us.anthropic.claude-sonnet-4-6"),
        ("Custom model ID", "custom"),
    ],
}

MODEL_OPTIONS["bedrock"] = _BEDROCK_MODELS
```

"Custom model ID" covers cross-region IDs (`apac.*`, `eu.*`), other
Anthropic versions, and any future non-Anthropic experiments.

## Validator

`tradingagents/llm_clients/validators.py`:

```python
if provider_lower in ("ollama", "openrouter", "bedrock"):
    return True
```

Reason: Bedrock model IDs vary by region and version suffix; precise
enumeration would either be incomplete or churn often. Treat them like
OpenRouter — accept any string, surface backend errors to the user.

## API key env mapping

`tradingagents/llm_clients/api_key_env.py`:

```python
"bedrock": "AWS_BEARER_TOKEN_BEDROCK",
```

The CLI's `ensure_api_key` will then prompt for `AWS_BEARER_TOKEN_BEDROCK`
when missing and persist it to `.env`. The primary user's shell already
exports it, so the prompt never fires for them — this is for Docker / CI /
fresh-machine bootstraps.

## CLI provider table

`cli/utils.py`, `_llm_provider_table`, after the Anthropic row:

```python
("AWS Bedrock", "bedrock", None),
```

`base_url` is `None` — Converse picks the endpoint from `region_name`, not
a URL. Leaving `backend_url` unset in `default_config` keeps that path
clean.

## `_get_provider_kwargs` change

`tradingagents/graph/trading_graph.py:138-156` gets a Bedrock branch that
reads the **same** `anthropic_effort` config key:

```python
elif provider == "bedrock":
    # Bedrock-on-Anthropic uses the same effort knob as direct Anthropic.
    effort = self.config.get("anthropic_effort")
    if effort:
        kwargs["effort"] = effort
```

Rationale: from the user's perspective, what runs on Bedrock is still Claude.
Inventing a separate `bedrock_effort` would force users to know two keys for
one concept. `temperature` already flows cross-provider via the shared
post-amble in the same function — Bedrock benefits automatically.

## `.env.example` block

Append to `.env.example`:

```
# AWS Bedrock (uses AWS_BEARER_TOKEN_BEDROCK + AWS_REGION from your shell)
#TRADINGAGENTS_LLM_PROVIDER=bedrock
#TRADINGAGENTS_DEEP_THINK_LLM=us.anthropic.claude-opus-4-7
#TRADINGAGENTS_QUICK_THINK_LLM=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

Commented out — Bedrock is opt-in.

## Tests

`tests/test_bedrock_client.py`, three unit tests:

```python
import pytest
from tradingagents.llm_clients import bedrock_client as mod


def _capture_kwargs(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        mod, "NormalizedChatBedrockConverse",
        lambda **kwargs: captured.setdefault("kwargs", kwargs),
    )
    return captured


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
```

No integration tests — they would burn Bedrock tokens, can't run in CI, and
would re-test what `langchain-aws` already covers. The three unit tests
verify the only logic that's *ours*: model passthrough, no `api_key` leak,
effort gating.

## Risks and mitigations

| Risk                                                         | Mitigation                                                |
| ------------------------------------------------------------ | --------------------------------------------------------- |
| `langchain-aws` upgrade changes Converse kwargs              | Unit tests fail loudly; pin floor `>=0.2.0`               |
| Bedrock token expires or region drifts                       | `ensure_api_key` prompt; `.env.example` documents `AWS_REGION` dependency |
| User enters Custom model ID without `us.` prefix             | Bedrock returns 400 with a clear message; we don't auto-correct |
| Effort field position changes in Converse spec               | Single-line edit in `bedrock_client.py` — `additional_model_request_fields` is an explicit escape hatch |
| `api_key` kwarg accidentally re-introduced via copy/paste    | Test `test_no_api_key_kwarg_leaked` guards against this  |

## Rollback

This is purely additive. To revert: delete `bedrock_client.py`, the
factory branch, the api_key_env entry, the model_catalog entry, the
validators tuple change, the trading_graph branch, the CLI table row,
the `.env.example` block, and the test file. No schema migrations, no
persisted state.

## Acceptance

After implementation, all of these must hold:

1. `.venv/bin/pytest tests/test_bedrock_client.py -v` — three tests pass.
2. `.venv/bin/pytest tests/ -m unit` — no regression in the existing unit
   suite (anthropic / openai / google clients still pass).
3. With `TRADINGAGENTS_LLM_PROVIDER=bedrock` and a `TRADINGAGENTS_DEEP_THINK_LLM`
   set, `tradingagents` CLI starts without prompting for an LLM provider or
   complaining about a missing key.
4. A short end-to-end run (one ticker) returns a real Bedrock response,
   driven by `AWS_BEARER_TOKEN_BEDROCK` from the user's shell.
