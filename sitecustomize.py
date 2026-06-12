import sys
import types


def _install_vllm_exceptions_shim() -> None:
    if "vllm.exceptions" in sys.modules:
        return

    try:
        import vllm.exceptions  # noqa: F401
        return
    except Exception:
        pass

    module = types.ModuleType("vllm.exceptions")

    class VLLMValidationError(ValueError):
        pass

    class VLLMNotFoundError(Exception):
        pass

    class LoRAAdapterNotFoundError(VLLMNotFoundError):
        def __init__(self, lora_name: str, lora_path: str) -> None:
            self.message = (
                f"Loading lora {lora_name} failed: "
                f"No adapter found for {lora_path}"
            )

        def __str__(self):
            return self.message

    module.VLLMValidationError = VLLMValidationError
    module.VLLMNotFoundError = VLLMNotFoundError
    module.LoRAAdapterNotFoundError = LoRAAdapterNotFoundError
    sys.modules["vllm.exceptions"] = module


def _patch_cross_attention_cache_blocks() -> None:
    try:
        from vllm.v1.core.single_type_kv_cache_manager import (
            CrossAttentionManager,
        )
    except Exception:
        return

    def _noop_cache_blocks(self, request, num_tokens, *args, **kwargs) -> None:
        return None

    CrossAttentionManager.cache_blocks = _noop_cache_blocks


def _register_bart_model() -> None:
    try:
        from vllm_bart_plugin import register_bart_model
    except Exception:
        return

    try:
        register_bart_model()
    except Exception:
        return


_install_vllm_exceptions_shim()
_register_bart_model()
_patch_cross_attention_cache_blocks()
