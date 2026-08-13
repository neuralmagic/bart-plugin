"""vLLM config hooks for BART-family models."""

from __future__ import annotations

from typing import TYPE_CHECKING

import vllm.config.vllm as vllm_config_module
from vllm.model_executor.models.config import (
    MODELS_CONFIG_MAP,
    VerifyAndUpdateConfig,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


BART_ARCHITECTURES = (
    "BartForConditionalGeneration",
    "Florence2ForConditionalGeneration",
)


class BartMRV2Config(VerifyAndUpdateConfig):
    """Require MRV2 for BART-family architectures."""

    @staticmethod
    def verify_and_update_config(vllm_config: "VllmConfig") -> None:
        if not vllm_config.use_v2_model_runner:
            raise ValueError(
                "BART-family models require the V2 model runner. "
                "Unset VLLM_USE_V2_MODEL_RUNNER or set it to 1."
            )


def register_bart_config() -> None:
    """Register BART-family config hooks and default MRV2 selection."""
    for architecture in BART_ARCHITECTURES:
        MODELS_CONFIG_MAP[architecture] = BartMRV2Config

    current = vllm_config_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
    vllm_config_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES = frozenset(
        (*current, *BART_ARCHITECTURES)
    )
    # The defaults are read through an lru_cache; drop any entry cached
    # before the plugin loaded so the new architectures take effect.
    vllm_config_module.default_v2_model_runner_architectures.cache_clear()
