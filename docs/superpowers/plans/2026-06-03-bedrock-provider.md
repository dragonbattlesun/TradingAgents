# AWS Bedrock LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `bedrock` LLM provider to TradingAgents so the framework can drive Anthropic Claude models on AWS Bedrock, reusing the `AWS_BEARER_TOKEN_BEDROCK` + `AWS_REGION` already exported in the user's shell.

**Architecture:** New `BedrockClient` parallel to `AnthropicClient`, both inheriting `BaseLLMClient`. Backed by `langchain_aws.ChatBedrockConverse` (Converse API for first-class tool use). Auth flows through boto3's standard credential chain — no `api_key` kwarg. The user's existing shell exports satisfy the entire credential picture.

**Tech Stack:** Python 3.13, `langchain-aws>=0.2.0` (new dependency), `langchain-core` (existing), `boto3` (transitively pulled in by `langchain-aws`), `pytest` for tests.

**Spec:** `docs/superpowers/specs/2026-06-03-bedrock-provider-design.md` (commit `ac5eff9`).

---

## File Structure

| File | Role |
|---|---|
| `tradingagents/llm_clients/bedrock_client.py` (new) | `BedrockClient` + `NormalizedChatBedrockConverse`. The only file that knows about `langchain-aws`. |
| `tradingagents/llm_clients/factory.py` (edit) | Add `bedrock` provider branch. Lazy import like other providers. |
| `tradingagents/llm_clients/api_key_env.py` (edit) | Map `bedrock` → `AWS_BEARER_TOKEN_BEDROCK` so the CLI's `ensure_api_key` works. |
| `tradingagents/llm_clients/model_catalog.py` (edit) | `MODEL_OPTIONS["bedrock"]` with 3 model IDs + Custom. |
| `tradingagents/llm_clients/validators.py` (edit) | Add `"bedrock"` to the skip-validation tuple. |
| `tradingagents/graph/trading_graph.py` (edit) | Mirror `anthropic_effort` in `_get_provider_kwargs` for the `bedrock` branch. |
| `cli/utils.py` (edit) | Add `("AWS Bedrock", "bedrock", None)` row to `_llm_provider_table`. |
| `pyproject.toml` (edit) | Add `langchain-aws>=0.2.0` to `dependencies`. |
| `.env.example` (edit) | Append commented-out Bedrock example block. |
| `tests/test_bedrock_client.py` (new) | Three unit tests pinning the only logic that's *ours*. |

Each task below is self-contained and ends with a commit. Run all `pytest`/`uv` commands from the repo root: `/Volumes/T7/github/TradingAgents`. Use the project's venv at `.venv/` (Python 3.13) — the executable is `.venv/bin/python` and `.venv/bin/pytest`.

**Important git note:** The repo's index has 144 macOS `._*` AppleDouble files staged from before this work began (visible in `git status`). They are noise. Every `git add` and `git commit` in this plan uses **explicit pathspecs** so we never accidentally include those files. Never run `git add .` or `git add -A`.

---

## Task 1: Add `langchain-aws` dependency

**Files:**
- Modify: `pyproject.toml` (the `dependencies` block, around line 11)

- [ ] **Step 1: Read current dependencies block**

Run: `grep -n langchain pyproject.toml`
Expected output includes lines like:
```
12:    "langchain-core>=0.3.81",
14:    "langchain-anthropic>=0.3.15",
15:    "langchain-experimental>=0.3.4",
16:    "langchain-google-genai>=4.0.0",
17:    "langchain-openai>=0.3.23",
```

- [ ] **Step 2: Add `langchain-aws>=0.2.0`**

Edit `pyproject.toml`. After the `langchain-anthropic` line and before `langchain-experimental`, add:
```
    "langchain-aws>=0.2.0",
```

The `dependencies` list stays alphabetized within the `langchain-*` cluster.

- [ ] **Step 3: Sync the venv with the new dependency**

Run: `VIRTUAL_ENV=$PWD/.venv .venv/bin/uv sync`
Expected: success message; `langchain-aws` and `boto3` appear in the install output.

- [ ] **Step 4: Verify import works**

Run: `.venv/bin/python -c "from langchain_aws import ChatBedrockConverse; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add -- pyproject.toml uv.lock
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
build: add langchain-aws dependency

Required for the upcoming AWS Bedrock provider, which drives Anthropic
Claude models via the Bedrock Converse API. boto3 comes in transitively.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- pyproject.toml uv.lock
```

Note: the explicit `--` pathspec at the end is mandatory — without it, the
144 staged `._*` AppleDouble files would be included.

---

## Task 2: Skip validator for `bedrock`

**Files:**
- Modify: `tradingagents/llm_clients/validators.py` (line 20)

This is a code-only change — no test in this task. The validator test will be added together with the rest of the test file in Task 3 (where the `bedrock_client` module also exists, so a single test file can be created cleanly without import-order gymnastics). The protection this skip-list change provides shows up *after* Task 5 adds `bedrock` to `MODEL_OPTIONS`: once `bedrock` is a known provider, `VALID_MODELS["bedrock"]` becomes a finite set, and only this skip-list entry keeps arbitrary IDs (custom region prefixes, future versions) from being rejected.

- [ ] **Step 1: Modify the skip-validation tuple**

Edit `tradingagents/llm_clients/validators.py:20`. Change:
```python
    if provider_lower in ("ollama", "openrouter"):
        return True
```
to:
```python
    if provider_lower in ("ollama", "openrouter", "bedrock"):
        return True
```

- [ ] **Step 2: Sanity-check the change**

Run:
```bash
.venv/bin/python -c "from tradingagents.llm_clients.validators import validate_model; print(validate_model('bedrock', 'us.anthropic.claude-opus-4-7'))"
```
Expected: `True`

- [ ] **Step 3: Run the existing unit suite for regressions**

Run: `.venv/bin/pytest tests/ -m unit -x -q 2>&1 | tail -10`
Expected: all existing unit tests pass. No regressions.

- [ ] **Step 4: Commit**

```bash
git add -- tradingagents/llm_clients/validators.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(llm): allow any model id for the bedrock provider

Bedrock model IDs vary by region prefix (us./apac./eu.) and version
suffix; precise enumeration would either be incomplete or churn often.
Treat them like OpenRouter — accept any string and surface backend
errors to the user.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- tradingagents/llm_clients/validators.py
```

---

## Task 3: Create `BedrockClient`

**Files:**
- Create: `tradingagents/llm_clients/bedrock_client.py`
- Create: `tests/test_bedrock_client.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bedrock_client.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bedrock_client.py -v 2>&1 | tail -20`
Expected: collection error — `ModuleNotFoundError: No module named 'tradingagents.llm_clients.bedrock_client'`

- [ ] **Step 3: Create the BedrockClient module**

Create `tradingagents/llm_clients/bedrock_client.py`:
```python
import os
import re
from typing import Any

from langchain_aws import ChatBedrockConverse

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_tokens", "temperature", "callbacks",
    # api_key / http_client intentionally NOT forwarded — auth comes
    # from boto3 (AWS_BEARER_TOKEN_BEDROCK), not a client kwarg.
)

# Bedrock model IDs may carry a region prefix: 'us.', 'apac.', 'eu.'.
_EFFORT_PATTERN = re.compile(
    r"^(?:[a-z]{2,3}\.)?anthropic\.claude-(opus|sonnet)-\d+-\d+"
)


def _supports_effort(model_id: str) -> bool:
    """Whether the Bedrock-hosted Claude model accepts the effort param.

    Mirrors the direct-Anthropic gate but matches Bedrock model IDs.
    Haiku does not support effort on either path; new Opus/Sonnet
    versions inherit support automatically via the regex.
    """
    return bool(_EFFORT_PATTERN.match(model_id.lower()))


class NormalizedChatBedrockConverse(ChatBedrockConverse):
    """ChatBedrockConverse with normalized content output."""

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class BedrockClient(BaseLLMClient):
    """Client for AWS Bedrock via the Converse API.

    Auth flows through boto3's standard credential chain. The convention
    this project uses is AWS_BEARER_TOKEN_BEDROCK + AWS_REGION (already
    exported in the user's shell for Claude Code). No api_key kwarg is
    accepted.
    """

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()

        region = (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
        )
        llm_kwargs: dict = {"model": self.model}
        if region:
            llm_kwargs["region_name"] = region

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Anthropic's "effort" maps to Converse's
        # additional_model_request_fields.
        effort = self.kwargs.get("effort")
        if effort and _supports_effort(self.model):
            llm_kwargs["additional_model_request_fields"] = {
                "thinking": {"type": effort},
            }

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("bedrock", self.model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_bedrock_client.py -v 2>&1 | tail -25`
Expected: 7 passed (1 validator + 2 client behavior tests + 4 parametrized effort tests).

- [ ] **Step 5: Run full unit suite for regressions**

Run: `.venv/bin/pytest tests/ -m unit -x -q 2>&1 | tail -10`
Expected: all unit tests pass. No regressions.

- [ ] **Step 6: Commit**

```bash
git add -- tradingagents/llm_clients/bedrock_client.py tests/test_bedrock_client.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(llm): add BedrockClient for AWS Bedrock Converse API

Drives Anthropic Claude models hosted on AWS Bedrock via
langchain-aws ChatBedrockConverse. Auth flows through boto3's
standard credential chain (AWS_BEARER_TOKEN_BEDROCK + AWS_REGION),
so no api_key kwarg is accepted. Effort gating mirrors the direct
Anthropic semantics — Haiku skips, Opus/Sonnet 4.5+ supported via
forward-compatible regex.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- tradingagents/llm_clients/bedrock_client.py tests/test_bedrock_client.py
```

---

## Task 4: Wire `bedrock` into the factory and api-key map

**Files:**
- Modify: `tradingagents/llm_clients/factory.py` (after line 47, the anthropic branch)
- Modify: `tradingagents/llm_clients/api_key_env.py` (PROVIDER_API_KEY_ENV dict, around line 17)

- [ ] **Step 1: Add a factory-routing test to the existing test file**

Append to `tests/test_bedrock_client.py`:
```python
@pytest.mark.unit
class TestBedrockFactoryWiring:
    def test_factory_returns_bedrock_client(self, monkeypatch):
        from tradingagents.llm_clients.factory import create_llm_client
        # Avoid actually instantiating ChatBedrockConverse — patch before
        # calling get_llm so no real boto3 client is built.
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
```

And:
```python
@pytest.mark.unit
class TestApiKeyEnvMapping:
    def test_bedrock_maps_to_aws_bearer_token(self):
        from tradingagents.llm_clients.api_key_env import get_api_key_env
        assert get_api_key_env("bedrock") == "AWS_BEARER_TOKEN_BEDROCK"
        # Case-insensitive — same convention used by other providers.
        assert get_api_key_env("BEDROCK") == "AWS_BEARER_TOKEN_BEDROCK"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bedrock_client.py::TestBedrockFactoryWiring tests/test_bedrock_client.py::TestApiKeyEnvMapping -v`
Expected: FAIL — `ValueError: Unsupported LLM provider: bedrock` for the factory test, and `assert None == 'AWS_BEARER_TOKEN_BEDROCK'` for the env map test.

- [ ] **Step 3: Add the factory branch**

Edit `tradingagents/llm_clients/factory.py`. After the `anthropic` block (lines 45-47):
```python
    if provider_lower == "anthropic":
        from .anthropic_client import AnthropicClient
        return AnthropicClient(model, base_url, **kwargs)
```
Add immediately after:
```python
    if provider_lower == "bedrock":
        from .bedrock_client import BedrockClient
        return BedrockClient(model, base_url, **kwargs)
```

- [ ] **Step 4: Add the api-key env mapping**

Edit `tradingagents/llm_clients/api_key_env.py`. In `PROVIDER_API_KEY_ENV` (after the `"anthropic"` line at 19):
```python
    "anthropic":  "ANTHROPIC_API_KEY",
```
Add immediately after:
```python
    "bedrock":    "AWS_BEARER_TOKEN_BEDROCK",
```

- [ ] **Step 5: Run the new tests**

Run: `.venv/bin/pytest tests/test_bedrock_client.py -v 2>&1 | tail -20`
Expected: 9 passed (7 from before + 2 new).

- [ ] **Step 6: Run full unit suite for regressions**

Run: `.venv/bin/pytest tests/ -m unit -x -q 2>&1 | tail -5`
Expected: all unit tests pass.

- [ ] **Step 7: Commit**

```bash
git add -- tradingagents/llm_clients/factory.py tradingagents/llm_clients/api_key_env.py tests/test_bedrock_client.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(llm): route bedrock provider through the factory

Adds the bedrock branch in create_llm_client and registers the
AWS_BEARER_TOKEN_BEDROCK env var so the CLI's ensure_api_key prompt
knows which variable to look for. Lazy import keeps langchain-aws
out of the import path for users who don't use Bedrock.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- tradingagents/llm_clients/factory.py tradingagents/llm_clients/api_key_env.py tests/test_bedrock_client.py
```

---

## Task 5: Add Bedrock model catalog entry

**Files:**
- Modify: `tradingagents/llm_clients/model_catalog.py` (add `_BEDROCK_MODELS`, register in `MODEL_OPTIONS`)

- [ ] **Step 1: Add a catalog test**

Append to `tests/test_bedrock_client.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bedrock_client.py::TestBedrockModelCatalog -v`
Expected: FAIL — `KeyError: 'bedrock'` in `MODEL_OPTIONS`.

- [ ] **Step 3: Add the Bedrock catalog block**

Edit `tradingagents/llm_clients/model_catalog.py`. Just before the `MODEL_OPTIONS` definition (around line 76), add:
```python
# Shared model list for AWS Bedrock — Anthropic Claude only.
# Bedrock model IDs carry a region prefix ("us.", "apac.", "eu.") that
# the Custom model ID escape hatch handles. Concrete entries use the
# us.* IDs that match what the user's shell already targets.
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
```

Then in `MODEL_OPTIONS`, after the `"anthropic": { ... }` block (around line 103), add:
```python
    # AWS Bedrock — Anthropic Claude only. Custom model ID escape hatch
    # covers cross-region IDs (apac.*, eu.*) and other versions.
    "bedrock": _BEDROCK_MODELS,
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_bedrock_client.py -v 2>&1 | tail -20`
Expected: 11 passed.

- [ ] **Step 5: Verify the validator skip-list still matters**

Run:
```bash
.venv/bin/python -c "
from tradingagents.llm_clients.validators import VALID_MODELS, validate_model
print('bedrock' in VALID_MODELS, 'should be True now that catalog has bedrock')
print(validate_model('bedrock', 'made-up-id'), 'should still be True via skip-list')
"
```
Expected output:
```
True should be True now that catalog has bedrock
True should still be True via skip-list
```

- [ ] **Step 6: Run the full unit suite for regressions**

Run: `.venv/bin/pytest tests/ -m unit -x -q 2>&1 | tail -5`
Expected: all unit tests pass.

- [ ] **Step 7: Commit**

```bash
git add -- tradingagents/llm_clients/model_catalog.py tests/test_bedrock_client.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(llm): add bedrock model catalog entries

Two concrete Anthropic Claude IDs per mode (Haiku/Sonnet for quick,
Opus/Sonnet for deep) plus the Custom model ID escape hatch covering
cross-region (apac./eu.) and future versions.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- tradingagents/llm_clients/model_catalog.py tests/test_bedrock_client.py
```

---

## Task 6: Pass `effort` through `_get_provider_kwargs`

**Files:**
- Modify: `tradingagents/graph/trading_graph.py` (around lines 153-156, after the `anthropic` branch)

This makes `config["anthropic_effort"]` apply to the `bedrock` provider too, so a single config knob covers both paths to Claude.

- [ ] **Step 1: Add a test for the kwargs branch**

Append to `tests/test_bedrock_client.py`:
```python
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
            config = {"llm_provider": "bedrock", "anthropic_effort": "high"}

        kwargs = method(Stub())
        assert kwargs.get("effort") == "high"

    def test_no_effort_when_unset(self):
        method = self._make_graph_method()

        class Stub:
            config = {"llm_provider": "bedrock"}

        kwargs = method(Stub())
        assert "effort" not in kwargs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_bedrock_client.py::TestProviderKwargsForBedrock -v`
Expected: FAIL on the first test — `assert kwargs.get("effort") == "high"` fails because `bedrock` has no branch yet.

- [ ] **Step 3: Add the bedrock branch in `_get_provider_kwargs`**

Edit `tradingagents/graph/trading_graph.py` at the end of the if/elif chain in `_get_provider_kwargs` (after the `anthropic` block at lines 153-156). The current shape:
```python
        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        # Sampling temperature is cross-provider: forward it whenever set.
```
Insert between the `anthropic` block and the `# Sampling temperature` comment:
```python
        elif provider == "bedrock":
            # Bedrock-on-Anthropic uses the same effort knob as direct
            # Anthropic — users see one config key for one concept.
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_bedrock_client.py::TestProviderKwargsForBedrock -v`
Expected: 2 passed.

- [ ] **Step 5: Run full unit suite for regressions**

Run: `.venv/bin/pytest tests/ -m unit -x -q 2>&1 | tail -5`
Expected: all unit tests pass.

- [ ] **Step 6: Commit**

```bash
git add -- tradingagents/graph/trading_graph.py tests/test_bedrock_client.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(graph): pass anthropic_effort through for bedrock provider

The user-facing knob is the same — Bedrock just routes Claude through
AWS — so reuse anthropic_effort instead of inventing a parallel
bedrock_effort. Keeps configuration surface area small.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- tradingagents/graph/trading_graph.py tests/test_bedrock_client.py
```

---

## Task 7: Add Bedrock to the CLI provider table

**Files:**
- Modify: `cli/utils.py` (`_llm_provider_table`, around lines 281-293)

- [ ] **Step 1: Add a CLI table test**

Append to `tests/test_bedrock_client.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_bedrock_client.py::TestCliProviderTable -v`
Expected: FAIL — `len(bedrock_rows) == 1` fails because no row exists yet.

- [ ] **Step 3: Add the table row**

Edit `cli/utils.py`. In `_llm_provider_table`, after the `Anthropic` row (line 284):
```python
        ("Anthropic", "anthropic", "https://api.anthropic.com/"),
```
Add immediately after:
```python
        ("AWS Bedrock", "bedrock", None),
```

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest tests/test_bedrock_client.py -v 2>&1 | tail -20`
Expected: 14 passed (so far across all classes).

- [ ] **Step 5: Run the full unit suite for regressions**

Run: `.venv/bin/pytest tests/ -m unit -x -q 2>&1 | tail -5`
Expected: all unit tests pass.

- [ ] **Step 6: Commit**

```bash
git add -- cli/utils.py tests/test_bedrock_client.py
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
feat(cli): add AWS Bedrock to the LLM provider picker

base_url is intentionally None — Bedrock's endpoint is region-derived
(passed as region_name to ChatBedrockConverse), not a URL the user
sets per call.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- cli/utils.py tests/test_bedrock_client.py
```

---

## Task 8: Document Bedrock in `.env.example`

**Files:**
- Modify: `.env.example` (append)

- [ ] **Step 1: Read current `.env.example`**

Run: `cat .env.example`
Expected: file ends with the `#TRADINGAGENTS_TEMPERATURE=0.0` line. Confirm before editing.

- [ ] **Step 2: Append the Bedrock block**

Use Edit to append at the end of `.env.example`:
```

# AWS Bedrock (uses AWS_BEARER_TOKEN_BEDROCK + AWS_REGION from your shell).
# Auth flows through boto3's standard credential chain — set the bearer
# token in your shell or via .env, no api_key needed. Bedrock model IDs
# carry a region prefix (us./apac./eu.) and may include a version suffix
# (e.g. -20251001-v1:0). The CLI accepts any string here; backend errors
# surface verbatim if the ID is wrong.
#TRADINGAGENTS_LLM_PROVIDER=bedrock
#TRADINGAGENTS_DEEP_THINK_LLM=us.anthropic.claude-opus-4-7
#TRADINGAGENTS_QUICK_THINK_LLM=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

- [ ] **Step 3: Verify**

Run: `tail -15 .env.example`
Expected: the new block is present and starts with a blank line separator.

- [ ] **Step 4: Commit**

```bash
git add -- .env.example
git -c commit.gpgsign=false commit -m "$(cat <<'EOF'
docs(env): document the bedrock provider in .env.example

Commented-out block — Bedrock is opt-in. Calls out the boto3 auth
chain (AWS_BEARER_TOKEN_BEDROCK + AWS_REGION) so users with the
existing Claude Code shell exports don't have to read code to know
their setup is sufficient.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)" -- .env.example
```

---

## Task 9: End-to-end smoke against the real Bedrock endpoint

**Files:** none modified.

This is a manual verification step — no code, no commit. It confirms acceptance criteria 3 and 4 from the spec. Skip only if you cannot reach AWS Bedrock (offline, no credentials, etc.) and report that gap to the user.

- [ ] **Step 1: Verify shell environment is set**

Run:
```bash
echo "AWS_REGION=$AWS_REGION"
echo "AWS_BEARER_TOKEN_BEDROCK length=${#AWS_BEARER_TOKEN_BEDROCK}"
```
Expected: `AWS_REGION=us-east-1` and a non-zero length (the token itself stays redacted in output).

If the length is 0, source the file: `source ~/.claude/bedrock.env` and re-check.

- [ ] **Step 2: Set the project env vars and probe `_apply_env_overrides`**

Run:
```bash
TRADINGAGENTS_LLM_PROVIDER=bedrock \
TRADINGAGENTS_DEEP_THINK_LLM=us.anthropic.claude-opus-4-7 \
TRADINGAGENTS_QUICK_THINK_LLM=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
.venv/bin/python -c "
from tradingagents.default_config import DEFAULT_CONFIG
print('provider:', DEFAULT_CONFIG['llm_provider'])
print('deep:    ', DEFAULT_CONFIG['deep_think_llm'])
print('quick:   ', DEFAULT_CONFIG['quick_think_llm'])
"
```
Expected:
```
provider: bedrock
deep:     us.anthropic.claude-opus-4-7
quick:    us.anthropic.claude-haiku-4-5-20251001-v1:0
```

- [ ] **Step 3: Send a single Bedrock invocation**

Run:
```bash
.venv/bin/python -c "
from tradingagents.llm_clients.factory import create_llm_client
from langchain_core.messages import HumanMessage
client = create_llm_client('bedrock', 'us.anthropic.claude-haiku-4-5-20251001-v1:0', max_tokens=64)
resp = client.get_llm().invoke([HumanMessage(content='Say hi in five words.')])
print(repr(resp.content))
"
```
Expected: a short string of ~5 words. If you see an `AccessDeniedException` or `botocore.exceptions.NoCredentialsError`, the shell environment is not picking up `AWS_BEARER_TOKEN_BEDROCK` — fix that and re-run rather than continuing.

- [ ] **Step 4: Optional: launch the CLI**

Run:
```bash
TRADINGAGENTS_LLM_PROVIDER=bedrock \
TRADINGAGENTS_DEEP_THINK_LLM=us.anthropic.claude-opus-4-7 \
TRADINGAGENTS_QUICK_THINK_LLM=us.anthropic.claude-haiku-4-5-20251001-v1:0 \
.venv/bin/tradingagents
```
Expected: the CLI **does not** prompt for an LLM provider or model (those steps are short-circuited because the env vars cover them) and **does not** complain about a missing API key. Cancel out before actually running an analysis if you don't want to burn tokens — the prompt for ticker / date is enough to confirm the wiring.

- [ ] **Step 5: Note any gaps**

If anything in steps 2-4 misbehaves, leave a comment block at the bottom of the spec file (`docs/superpowers/specs/2026-06-03-bedrock-provider-design.md`) under a `## Post-implementation notes` heading and report back to the user. **Don't silently fix without surfacing the issue.**

---

## Final acceptance check

- [ ] **Step 1: Run the full unit suite once more**

Run: `.venv/bin/pytest tests/ -m unit -q 2>&1 | tail -5`
Expected: all unit tests pass, including the 14+ new ones in `test_bedrock_client.py`.

- [ ] **Step 2: Verify no `._*` files were committed**

Run: `git log -p --since='1 hour ago' -- '._*' 2>&1 | head`
Expected: empty output (no AppleDouble files in any of the new commits).

- [ ] **Step 3: Sanity-check the commit list**

Run: `git log --oneline -10`
Expected: ~8 new commits on top of `04f434e`, all with explicit pathspecs and Co-Authored-By trailers. Each commit's diff scope matches its task.

If everything passes, the implementation is done. Report status to the user with the commit list.
