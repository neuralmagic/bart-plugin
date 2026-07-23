"""vLLM BART model plugin.

This plugin registers the BART model with vLLM's ModelRegistry,
allowing it to be used with vLLM's inference engine.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.model_executor.models.registry import ModelRegistry

__version__ = "0.1.0"


def register_bart_model() -> None:
    """Register BART models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    try:
        from vllm.logger import init_logger
        from vllm.model_executor.models.registry import ModelRegistry

        logger = init_logger(__name__)
        # Register BartForConditionalGeneration with the ModelRegistry
        # Using lazy loading to avoid importing the model class during plugin discovery
        ModelRegistry.register_model(
            "BartForConditionalGeneration",
            "vllm_bart_plugin.bart:BartForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "Florence2ForConditionalGeneration",
            "vllm_bart_plugin.florence2:Florence2ForConditionalGeneration",
        )
        ModelRegistry.register_model(
            "T5Gemma2ForConditionalGeneration",
            "vllm_bart_plugin.t5gemma2:T5Gemma2ForConditionalGeneration",
        )

        logger.info("Successfully registered BART model with vLLM")

    except Exception as e:
        logger.error(f"Failed to register BART model: {e}")
        raise


__all__ = [
    "register_bart_model",
    "__version__",
]
