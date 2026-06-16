"""vLLM BART model plugin.

This plugin registers the BART model with vLLM's ModelRegistry,
allowing it to be used with vLLM's inference engine.
"""

import os
import sys

__version__ = "0.1.0"

_MODEL_REGISTRATIONS = (
    (
        "BartForConditionalGeneration",
        "vllm_bart_plugin.bart:BartForConditionalGeneration",
    ),
    (
        "Florence2ForConditionalGeneration",
        "vllm_bart_plugin.florence2:Florence2ForConditionalGeneration",
    ),
)


def _clear_vllm_env_cache() -> None:
    envs = sys.modules.get("vllm.envs")
    env_getattr = getattr(envs, "__getattr__", None) if envs is not None else None
    cache_clear = getattr(env_getattr, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


def force_mrv2_model_runner() -> None:
    """BART-family models require MRV2."""
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    _clear_vllm_env_cache()


def register_bart_model() -> None:
    """Register BART models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    force_mrv2_model_runner()

    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        for model_name, model_ref in _MODEL_REGISTRATIONS:
            ModelRegistry.register_model(model_name, model_ref)

        from vllm_bart_plugin.openai_serving import install_openai_prompt_adapter

        install_openai_prompt_adapter()

        logger.info("Successfully registered BART model with vLLM")

    except Exception as e:
        logger.error(f"Failed to register BART model: {e}")
        raise


__all__ = [
    "force_mrv2_model_runner",
    "register_bart_model",
    "__version__",
]
