"""Tests for OpenAI completion prompt adaptation."""

import asyncio
from types import SimpleNamespace

from vllm_bart_plugin.openai_serving import (
    _is_bart_model,
    _patch_preprocess_completion,
    _wrap_prompt_input,
)


class FakeTokenizer:
    bos_token_id = 0

    def decode(self, token_ids):
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


def test_wrap_text_prompt():
    assert _wrap_prompt_input(FakeTokenizer(), "summarize this") == {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {"text": "summarize this"},
        },
        "decoder_prompt": {"prompt_token_ids": [0]},
    }


def test_wrap_token_ids_decodes():
    wrapped = _wrap_prompt_input(FakeTokenizer(), [1, 2, 3])

    assert wrapped["encoder_prompt"]["multi_modal_data"] == {
        "text": "decoded:1,2,3"
    }


def test_wrap_batch_of_prompts():
    wrapped = _wrap_prompt_input(FakeTokenizer(), ["a", [1, 2]])

    assert [w["encoder_prompt"]["multi_modal_data"]["text"] for w in wrapped] == [
        "a",
        "decoded:1,2",
    ]


def test_wrap_without_bos_token_is_a_passthrough():
    tokenizer = SimpleNamespace(bos_token_id=None)

    assert _wrap_prompt_input(tokenizer, "text") == "text"


def test_is_bart_model_matches_architecture_fallback():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=""),
        architectures=["Florence2ForConditionalGeneration"],
    )

    assert _is_bart_model(model_config)


def test_patch_preprocess_completion_wraps_prompt():
    class FakeServing:
        def __init__(self):
            self.model_config = SimpleNamespace(
                hf_config=SimpleNamespace(model_type="bart"),
            )
            self.renderer = SimpleNamespace(tokenizer=FakeTokenizer())

        async def preprocess_completion(
            self,
            request,
            prompt_input,
            prompt_embeds,
            *,
            skip_mm_cache=False,
        ):
            return prompt_input

    _patch_preprocess_completion(FakeServing)

    wrapped = asyncio.run(
        FakeServing().preprocess_completion(None, "summarize this", None)
    )

    assert wrapped["encoder_prompt"]["multi_modal_data"] == {
        "text": "summarize this"
    }
    assert wrapped["decoder_prompt"] == {"prompt_token_ids": [0]}
