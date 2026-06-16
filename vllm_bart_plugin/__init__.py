"""vLLM BART model plugin.

This plugin registers the BART model with vLLM's ModelRegistry,
allowing it to be used with vLLM's inference engine.
"""

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


def register_bart_model() -> None:
    """Register BART models with vLLM's ModelRegistry.

    This function is called automatically when the plugin is loaded
    through vLLM's plugin discovery mechanism.
    """
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    try:
        from vllm.model_executor.models.registry import ModelRegistry

        from vllm_bart_plugin.config import register_bart_config

        register_bart_config()

        for model_name, model_ref in _MODEL_REGISTRATIONS:
            ModelRegistry.register_model(model_name, model_ref)

        from vllm_bart_plugin.openai_serving import install_openai_prompt_adapter

        install_openai_prompt_adapter()

        logger.info("Successfully registered BART model with vLLM")

    except Exception as e:
        logger.error(f"Failed to register BART model: {e}")
        raise


__all__ = [
    "register_bart_model",
    "__version__",
]
