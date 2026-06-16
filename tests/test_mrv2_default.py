"""Tests for BART MRV2 config defaults."""

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
    for architecture in BART_ARCHITECTURES:
        monkeypatch.delitem(MODELS_CONFIG_MAP, architecture, raising=False)

    register_bart_config()

    for architecture in BART_ARCHITECTURES:
        assert MODELS_CONFIG_MAP[architecture] is BartMRV2Config
        assert (
            architecture
            in vllm_config_module.DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES
        )


def test_bart_config_rejects_explicit_mrv2_disable(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    with pytest.raises(ValueError, match="require the V2 model runner"):
        BartMRV2Config.verify_and_update_config(None)
