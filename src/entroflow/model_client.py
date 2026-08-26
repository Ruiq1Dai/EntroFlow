"""AutoGen model client configured through environment variables."""

from __future__ import annotations

import os


def create_model_client():
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    required = ("MODEL_API_KEY", "MODEL_BASE_URL", "MODEL_NAME")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"missing model environment variables: {', '.join(missing)}")
    return OpenAIChatCompletionClient(
        model=os.environ["MODEL_NAME"],
        api_key=os.environ["MODEL_API_KEY"],
        base_url=os.environ["MODEL_BASE_URL"],
        model_info={
            "function_calling": False,
            "json_output": False,
            "vision": False,
            "family": "unknown",
            "structured_output": False,
        },
        parallel_tool_calls=False,
        timeout=120.0,
        max_retries=0,
    )
