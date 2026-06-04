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
# The trailing `(?:-\S+)?$` tolerates dated/versioned suffixes such as
# `-20251001-v1:0` while still anchoring the end so a malformed id like
# `claude-sonnet-4-6-haiku-variant` cannot slip through.
_EFFORT_PATTERN = re.compile(
    r"^(?:[a-z]{2,3}\.)?anthropic\.claude-(opus|sonnet)-\d+-\d+(?:-\S+)?$"
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

        # self.base_url is intentionally ignored — Bedrock endpoints are
        # region-derived (region_name kwarg), not URL-configurable.
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

        # Anthropic's top-level "effort" param has no direct equivalent
        # on Bedrock Converse — the schema splits it into two fields
        # under additional_model_request_fields:
        #   thinking.type      -> "adaptive" turns extended thinking on
        #                         (4.7+ rejects the older "enabled")
        #   output_config.effort -> "high" / "medium" / "low" depth knob
        # Bedrock's own error reply spells this mapping out:
        #   '"thinking.type.enabled" is not supported for this model.
        #    Use "thinking.type.adaptive" and "output_config.effort"'.
        effort = self.kwargs.get("effort")
        if effort and _supports_effort(self.model):
            llm_kwargs["additional_model_request_fields"] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            }

        return NormalizedChatBedrockConverse(**llm_kwargs)

    def validate_model(self) -> bool:
        return validate_model("bedrock", self.model)
