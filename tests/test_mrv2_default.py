"""Tests for BART MRV2 runner defaults."""

import os
import sys
import types

from vllm_bart_plugin import force_mrv2_model_runner


def test_force_mrv2_model_runner_sets_vllm_env(monkeypatch):
    monkeypatch.delenv("VLLM_BART_FORCE_MRV2", raising=False)
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    force_mrv2_model_runner()

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "1"


def test_force_mrv2_model_runner_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VLLM_BART_FORCE_MRV2", "0")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")

    force_mrv2_model_runner()

    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_force_mrv2_model_runner_clears_vllm_env_cache(monkeypatch):
    cleared = False

    def fake_getattr(_name):
        raise AttributeError

    def cache_clear():
        nonlocal cleared
        cleared = True

    fake_getattr.cache_clear = cache_clear
    fake_envs = types.ModuleType("vllm.envs")
    fake_envs.__getattr__ = fake_getattr

    monkeypatch.delenv("VLLM_BART_FORCE_MRV2", raising=False)
    monkeypatch.setitem(sys.modules, "vllm.envs", fake_envs)

    force_mrv2_model_runner()

    assert cleared
