"""Tests for BART MRV2 config defaults."""

from types import SimpleNamespace

import pytest
import vllm.config.vllm as vllm_config_module
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from vllm_bart_plugin.config import (
    BART_ARCHITECTURES,
    BartMRV2Config,
    register_bart_config,
)


def test_register_bart_config_marks_architectures_as_default_mrv2(monkeypatch):
    monkeypatch.setattr(
        vllm_config_module,
        "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES",
        frozenset({"LlamaForCausalLM"}),
    )
    saved = {
        architecture: MODELS_CONFIG_MAP.pop(architecture, None)
        for architecture in BART_ARCHITECTURES
    }
    try:
        register_bart_config()

        for architecture in BART_ARCHITECTURES:
            assert MODELS_CONFIG_MAP[architecture] is BartMRV2Config
            assert (
                architecture in vllm_config_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
            )
    finally:
        for architecture, config_cls in saved.items():
            if config_cls is None:
                MODELS_CONFIG_MAP.pop(architecture, None)
            else:
                MODELS_CONFIG_MAP[architecture] = config_cls
        defaults = getattr(
            vllm_config_module,
            "default_v2_model_runner_architectures",
            None,
        )
        if defaults is not None:
            defaults.cache_clear()


def test_register_bart_config_supports_uncached_mrv2_defaults(monkeypatch):
    monkeypatch.setattr(
        vllm_config_module,
        "DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES",
        vllm_config_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES,
    )
    monkeypatch.delattr(
        vllm_config_module,
        "default_v2_model_runner_architectures",
        raising=False,
    )
    saved = {
        architecture: MODELS_CONFIG_MAP.pop(architecture, None)
        for architecture in BART_ARCHITECTURES
    }
    try:
        register_bart_config()

        for architecture in BART_ARCHITECTURES:
            assert architecture in (
                vllm_config_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
            )
    finally:
        for architecture, config_cls in saved.items():
            if config_cls is None:
                MODELS_CONFIG_MAP.pop(architecture, None)
            else:
                MODELS_CONFIG_MAP[architecture] = config_cls


def test_bart_config_rejects_non_mrv2_runner():
    with pytest.raises(ValueError, match="require the V2 model runner"):
        BartMRV2Config.verify_and_update_config(
            SimpleNamespace(use_v2_model_runner=False)
        )


def test_bart_config_accepts_mrv2_runner():
    BartMRV2Config.verify_and_update_config(SimpleNamespace(use_v2_model_runner=True))
