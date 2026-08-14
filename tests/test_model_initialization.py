"""Tests for BART model initialization."""

import pytest
import torch
from vllm import LLM


def _register_bart_plugin():
    from vllm_bart_plugin import register_bart_model

    register_bart_model()


class TestModelInitialization:
    """Test BART model initialization with vLLM."""

    def test_encoder_layer_always_clamps_fp16(self, monkeypatch):
        """Keep the FP16 overflow guard on-device without a scalar read."""
        from vllm_bart_plugin import bart

        layer = bart.BartEncoderLayer.__new__(bart.BartEncoderLayer)
        torch.nn.Module.__init__(layer)

        def identity(hidden_states):
            return hidden_states

        layer.__dict__.update(
            self_attn=lambda *, hidden_states: torch.zeros_like(hidden_states),
            self_attn_layer_norm=identity,
            activation_fn=identity,
            fc1=lambda hidden_states: (hidden_states, None),
            fc2=lambda hidden_states: (hidden_states, None),
            final_layer_norm=identity,
        )

        clamp_calls = []
        original_clamp = bart.cast_overflow_tensors

        def track_clamp(hidden_states):
            clamp_calls.append(hidden_states)
            return original_clamp(hidden_states)

        monkeypatch.setattr(bart, "cast_overflow_tensors", track_clamp)
        output = layer(torch.ones((2, 4), dtype=torch.float16))

        assert len(clamp_calls) == 1
        assert output.dtype == torch.float16

        layer(torch.ones((2, 4), dtype=torch.float32))
        assert len(clamp_calls) == 1

    @pytest.mark.slow
    def test_model_loads(self, small_model_name):
        """Test that BART model can be loaded."""
        try:
            llm = LLM(
                model=small_model_name,
                trust_remote_code=False,
                dtype="float16",
                enforce_eager=True,
                max_model_len=512,
                gpu_memory_utilization=0.3,
            )
            assert llm is not None
        except Exception as e:
            pytest.fail(f"Failed to load model: {e}")

    @pytest.mark.slow
    def test_model_with_custom_config(self, small_model_name):
        """Test BART model with custom configuration."""
        try:
            llm = LLM(
                model=small_model_name,
                trust_remote_code=False,
                dtype="float16",
                tensor_parallel_size=1,
                max_num_seqs=2,
                max_num_batched_tokens=4096,
                gpu_memory_utilization=0.3,
                enforce_eager=True,
            )
            assert llm is not None
        except Exception as e:
            pytest.fail(f"Failed to load model with config: {e}")

    @pytest.mark.slow
    def test_model_class_initialization(self):
        """Test that model class can be instantiated."""
        _register_bart_plugin()

        from vllm_bart_plugin.bart import BartForConditionalGeneration
        from transformers import BartConfig
        from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig

        # Create minimal config
        hf_config = BartConfig.from_pretrained("facebook/bart-large-cnn")

        model_config = ModelConfig(
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
        )
        model_config.hf_config = hf_config

        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.3,
            cache_dtype="auto",
        )

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            load_config=LoadConfig(),
        )

        # Try to instantiate the model
        try:
            model = BartForConditionalGeneration(vllm_config=vllm_config)
            assert model is not None
            assert hasattr(model, "model")
            assert hasattr(model, "lm_head")
        except Exception as e:
            pytest.fail(f"Failed to instantiate model: {e}")

    def test_model_has_required_methods(self):
        """Test that model has required methods."""
        from vllm_bart_plugin.bart import BartForConditionalGeneration

        required_methods = [
            "forward",
            "compute_logits",
            "load_weights",
            "embed_multimodal",
        ]

        for method in required_methods:
            assert hasattr(BartForConditionalGeneration, method), (
                f"Model missing required method: {method}"
            )

    @pytest.mark.slow
    def test_encoder_decoder_structure(self):
        """Test that BART has proper encoder-decoder structure."""
        _register_bart_plugin()

        from vllm_bart_plugin.bart import BartModel, BartEncoder, BartDecoder
        from transformers import BartConfig
        from vllm.config import CacheConfig, LoadConfig, ModelConfig, VllmConfig

        hf_config = BartConfig.from_pretrained("facebook/bart-large-cnn")

        model_config = ModelConfig(
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn",
            tokenizer_mode="auto",
            trust_remote_code=False,
            dtype="float16",
            seed=0,
        )
        model_config.hf_config = hf_config

        cache_config = CacheConfig(
            block_size=16,
            gpu_memory_utilization=0.3,
            cache_dtype="auto",
        )

        vllm_config = VllmConfig(
            model_config=model_config,
            cache_config=cache_config,
            load_config=LoadConfig(),
        )

        model = BartModel(vllm_config=vllm_config)

        assert hasattr(model, "encoder")
        assert hasattr(model, "decoder")
        assert isinstance(model.encoder, BartEncoder)
        assert isinstance(model.decoder, BartDecoder)
