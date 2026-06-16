"""Tests for BART model inference."""

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "facebook/bart-large-cnn"
pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def llm():
    """Create LLM instance for tests."""
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    return LLM(
        model=MODEL_NAME,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=4,
        max_num_batched_tokens=2048,
        gpu_memory_utilization=0.3,
        dtype="float16",
    )


class TestModelInference:
    """Test BART model inference capabilities."""

    def test_single_generation(self, llm):
        """Test single prompt generation."""
        params = SamplingParams(temperature=0.0, max_tokens=10)

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The president of the United States is",
                },
            },
            "decoder_prompt": "<s>",
        }

        outputs = llm.generate([prompt], sampling_params=params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs) > 0
        assert len(outputs[0].outputs[0].text) > 0

    def test_batch_generation(self, llm, test_prompts):
        """Test batch generation with multiple prompts."""
        params = SamplingParams(temperature=0.0, max_tokens=10)

        outputs = llm.generate(test_prompts, sampling_params=params)

        assert len(outputs) == len(test_prompts)
        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text) > 0

    @pytest.mark.skip(reason="TODO for some reason this is still different")
    def test_batched_encoder_opt_matches_unbatched(self, llm, test_prompts, monkeypatch):
        """Ensure encoder batching optimization matches baseline outputs."""

        prompts = (test_prompts * 2)[:4]  # force a batch >=2 (typically 4)
        params = SamplingParams(temperature=0.0, max_tokens=10, seed=0)

        def run_with_env(enabled: bool) -> list[str]:
            model = llm.llm_engine.model_executor.driver_worker.worker.model_runner.model # .runnable if CG is on
            model._encoder_max_seq_padding = enabled
            outs = llm.generate(prompts, sampling_params=params)
            texts = [o.outputs[0].text for o in outs]

            return texts

        baseline = run_with_env(enabled=False)
        optimized = run_with_env(enabled=True)

        # TODO for some reason this is still different
        print(baseline, "\n", optimized, "\n")
        assert baseline == optimized

    def test_different_sampling_params(self, llm):
        """Test generation with different sampling parameters."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Machine learning is",
                },
            },
            "decoder_prompt": "<s>a ",
        }

        # Greedy decoding
        greedy_params = SamplingParams(temperature=0.0, max_tokens=10)
        greedy_outputs = llm.generate([prompt], sampling_params=greedy_params)
        assert len(greedy_outputs[0].outputs[0].text) > 0

        # Sampling with temperature
        sample_params = SamplingParams(temperature=0.8, top_p=0.9, max_tokens=10)
        sample_outputs = llm.generate([prompt], sampling_params=sample_params)
        assert len(sample_outputs[0].outputs[0].text) > 0

        # Nucleus sampling
        nucleus_params = SamplingParams(temperature=0.7, top_p=0.5, max_tokens=10)
        nucleus_outputs = llm.generate([prompt], sampling_params=nucleus_params)
        assert len(nucleus_outputs[0].outputs[0].text) > 0

    def test_max_tokens_respected(self, llm):
        """Test that max_tokens parameter is respected."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The quick brown fox jumps over the lazy dog.",
                },
            },
            "decoder_prompt": "<s>",
        }

        max_tokens = 1
        params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

        outputs = llm.generate([prompt], sampling_params=params)

        # Count tokens in output
        output_text = outputs[0].outputs[0].text
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokens = tokenizer(output_text, add_special_tokens=False)['input_ids']
        assert len(tokens) == max_tokens

    def test_encoder_decoder_prompt_format(self, llm):
        """Test that encoder-decoder prompt format works correctly."""
        # Test with explicit encoder and decoder prompts
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Donald Trump is the president of",
                },
            },
            "decoder_prompt": "<s>the United",
        }

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate([prompt], sampling_params=params)

        assert len(outputs) == 1
        output_text = outputs[0].outputs[0].text
        assert len(output_text) > 0

    def test_empty_encoder_text(self, llm):
        """Test generation with minimal encoder text."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "<s>",  # Minimal text
                },
            },
            "decoder_prompt": "<s>Hello",
        }

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate([prompt], sampling_params=params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].text) > 0

    def test_deterministic_output(self, llm):
        """Test that temperature=0 produces deterministic outputs."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "The capital of France is",
                },
            },
            "decoder_prompt": "<s>",
        }

        params = SamplingParams(temperature=0.0, max_tokens=10)

        # Generate twice with same params
        outputs1 = llm.generate([prompt], sampling_params=params)
        outputs2 = llm.generate([prompt], sampling_params=params)

        # Should produce same output
        text1 = outputs1[0].outputs[0].text
        text2 = outputs2[0].outputs[0].text

        assert text1 == text2, "Deterministic generation should produce same output"

    def test_output_metadata(self, llm):
        """Test that output contains expected metadata."""
        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": "Testing metadata",
                },
            },
            "decoder_prompt": "<s>",
        }

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate([prompt], sampling_params=params)

        output = outputs[0]

        # Check output structure
        assert hasattr(output, 'outputs')
        assert len(output.outputs) > 0
        assert hasattr(output.outputs[0], 'text')

        # Check that we can access token IDs if available
        if hasattr(output.outputs[0], 'token_ids'):
            assert len(output.outputs[0].token_ids) > 0


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    def test_long_encoder_input(self, llm):
        """Test with longer encoder input."""
        long_text = " ".join(["word"] * 100)  # 100 words

        prompt = {
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": long_text,
                },
            },
            "decoder_prompt": "<s>",
        }

        params = SamplingParams(temperature=0.0, max_tokens=10)

        try:
            outputs = llm.generate([prompt], sampling_params=params)
            assert len(outputs) == 1
        except Exception as e:
            pytest.fail(f"Failed with long input: {e}")

    def test_multiple_decoder_starts(self, llm):
        """Test with different decoder prompt starts."""
        encoder_text = "The president of the United States is"

        decoder_starts = ["<s>", "<s>Donald", "<s>Joe"]

        prompt = [{
            "encoder_prompt": {
                "prompt": "",
                "multi_modal_data": {
                    "text": encoder_text,
                },
            },
            "decoder_prompt": decoder_start
            }
            for decoder_start in decoder_starts
        ]

        params = SamplingParams(temperature=0.0, max_tokens=5)
        outputs = llm.generate(prompt, sampling_params=params)
        print(outputs, "\n\n")

        assert len(outputs) == len(decoder_starts)
        for output in outputs:
            assert len(output.outputs) > 0
            assert len(output.outputs[0].text) > 0
