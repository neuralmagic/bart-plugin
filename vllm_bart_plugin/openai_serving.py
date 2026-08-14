"""Adapt OpenAI completion prompts for BART-family models.

vLLM's completions path treats the request prompt as decoder input, while
BART-family models consume encoder text through ``multi_modal_data``. Wrap
plain prompts into explicit encoder/decoder prompts so standard OpenAI
clients keep working.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger

from vllm_bart_plugin.config import BART_ARCHITECTURES

logger = init_logger(__name__)

_BART_MODEL_TYPES = {"bart", "mbart", "florence2"}


def _is_bart_model(model_config: Any) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    if model_type in _BART_MODEL_TYPES:
        return True
    architectures = getattr(model_config, "architectures", None) or ()
    return any(arch in BART_ARCHITECTURES for arch in architectures)


def _wrap_prompt(tokenizer: Any, prompt: str | list[int]) -> dict[str, Any]:
    text = prompt if isinstance(prompt, str) else tokenizer.decode(prompt)
    return {
        "encoder_prompt": {"prompt": "", "multi_modal_data": {"text": text}},
        "decoder_prompt": {"prompt_token_ids": [int(tokenizer.bos_token_id)]},
    }


def _wrap_prompt_input(tokenizer: Any, prompt_input: Any) -> Any:
    """Wrap a completion prompt (str | list[str] | list[int] | list[list[int]])."""
    if getattr(tokenizer, "bos_token_id", None) is None:
        return prompt_input
    if isinstance(prompt_input, str):
        return _wrap_prompt(tokenizer, prompt_input)
    if isinstance(prompt_input, list) and prompt_input:
        if all(isinstance(token_id, int) for token_id in prompt_input):
            return _wrap_prompt(tokenizer, prompt_input)
        return [_wrap_prompt_input(tokenizer, prompt) for prompt in prompt_input]
    return prompt_input


def _patch_preprocess_completion(serving_cls: type[Any]) -> None:
    if vars(serving_cls).get("_vllm_bart_prompt_patched", False):
        return

    original_preprocess_completion = serving_cls.preprocess_completion

    async def patched_preprocess_completion(
        self,
        request,
        prompt_input,
        prompt_embeds=None,
        *args,
        **kwargs,
    ):
        if prompt_embeds is None and _is_bart_model(self.model_config):
            prompt_input = _wrap_prompt_input(self.renderer.tokenizer, prompt_input)
        return await original_preprocess_completion(
            self,
            request,
            prompt_input,
            prompt_embeds,
            *args,
            **kwargs,
        )

    serving_cls.preprocess_completion = patched_preprocess_completion
    serving_cls._vllm_bart_prompt_patched = True


def install_openai_prompt_adapter() -> None:
    try:
        from vllm.renderers.online_renderer import OnlineRenderer
    except ImportError:
        logger.warning(
            "Could not import vllm.renderers.online_renderer.OnlineRenderer; "
            "the OpenAI completion prompt adapter is not installed. Plain "
            "/v1/completions prompts will not be routed to the BART encoder."
        )
        return
    _patch_preprocess_completion(OnlineRenderer)
