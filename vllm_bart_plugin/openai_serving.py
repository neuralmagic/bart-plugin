from __future__ import annotations

import hashlib
import importlib
from typing import Any

_BART_MODEL_TYPES = {"bart", "mbart", "florence2"}


def _is_bart_family_model(model_config: Any) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    architectures = getattr(model_config, "architectures", None)
    if not architectures and hf_config is not None:
        architectures = getattr(hf_config, "architectures", None)
    arch_text = " ".join(str(arch) for arch in (architectures or ())).lower()
    return (
        model_type in _BART_MODEL_TYPES
        or "bart" in arch_text
        or "florence2" in arch_text
    )


def _is_encoder_decoder_model(model_config: Any) -> bool:
    if getattr(model_config, "is_encoder_decoder", False):
        return True
    hf_config = getattr(model_config, "hf_config", None)
    return bool(getattr(hf_config, "is_encoder_decoder", False))


def _encoder_cache_salt(
    prompt_text: str,
    existing_salt: str | None = None,
) -> str | None:
    if existing_salt:
        return existing_salt
    if not prompt_text:
        return None
    digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return f"bart-enc:{digest}"


def _decoder_bos_prompt(tokenizer: Any) -> dict[str, list[int]] | None:
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None:
        return None
    return {"prompt_token_ids": [int(bos)]}


def _extract_prompt_parts(
    tokenizer: Any,
    engine_prompt: Any,
) -> tuple[str | None, str | None, Any]:
    if isinstance(engine_prompt, str):
        return engine_prompt, None, None

    if isinstance(engine_prompt, list):
        if tokenizer is None:
            return None, None, None
        return tokenizer.decode(engine_prompt), None, None

    if not isinstance(engine_prompt, dict):
        return None, None, None

    prompt_text = engine_prompt.get("prompt")
    if prompt_text is None:
        prompt_token_ids = engine_prompt.get("prompt_token_ids")
        if prompt_token_ids is not None and tokenizer is not None:
            prompt_text = tokenizer.decode(prompt_token_ids)
    return (
        prompt_text,
        engine_prompt.get("cache_salt"),
        engine_prompt.get("mm_processor_kwargs"),
    )


def _wrap_completion_prompt(tokenizer: Any, engine_prompt: Any) -> Any:
    if isinstance(engine_prompt, dict) and "encoder_prompt" in engine_prompt:
        return engine_prompt

    decoder_prompt = _decoder_bos_prompt(tokenizer)
    if decoder_prompt is None:
        return engine_prompt

    prompt_text, cache_salt, mm_processor_kwargs = _extract_prompt_parts(
        tokenizer,
        engine_prompt,
    )
    if (
        isinstance(engine_prompt, dict)
        and engine_prompt.get("multi_modal_data") is not None
    ):
        wrapped_prompt = dict(engine_prompt)
    elif prompt_text is None:
        return engine_prompt
    else:
        wrapped_prompt = {
            "prompt": "",
            "multi_modal_data": {"text": prompt_text},
        }
    if mm_processor_kwargs is not None:
        wrapped_prompt["mm_processor_kwargs"] = mm_processor_kwargs
    if cache_salt := _encoder_cache_salt(prompt_text, cache_salt):
        wrapped_prompt["cache_salt"] = cache_salt
    return {
        "encoder_prompt": wrapped_prompt,
        "decoder_prompt": decoder_prompt,
    }


def _wrap_prompt_input(tokenizer: Any, prompt_input: Any) -> Any:
    if prompt_input is None:
        return None
    if isinstance(prompt_input, (str, dict)):
        return _wrap_completion_prompt(tokenizer, prompt_input)
    if isinstance(prompt_input, list):
        if not prompt_input:
            return prompt_input
        if all(isinstance(token_id, int) for token_id in prompt_input):
            return _wrap_completion_prompt(tokenizer, prompt_input)
        return [_wrap_completion_prompt(tokenizer, prompt) for prompt in prompt_input]
    return prompt_input


def _patch_preprocess_completion(serving_cls: type[Any]) -> None:
    # vars() rather than getattr: an inherited flag must not skip a subclass
    # that overrides preprocess_completion.
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
        if (
            prompt_embeds is None
            and _is_encoder_decoder_model(self.model_config)
            and _is_bart_family_model(self.model_config)
        ):
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
    for module_name, class_name in (
        ("vllm.entrypoints.serve.render.serving", "OpenAIServingRender"),
        ("vllm.renderers.online_renderer", "OnlineRenderer"),
    ):
        try:
            module = importlib.import_module(module_name)
            serving_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
        _patch_preprocess_completion(serving_cls)
