"""Tests for OpenAI completion prompt adaptation."""

from types import SimpleNamespace

from vllm_bart_plugin.openai_serving import (
    _is_bart_family_model,
    _wrap_completion_prompt,
)


class FakeTokenizer:
    bos_token_id = 0

    def decode(self, token_ids):
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)


def test_wrap_completion_prompt_from_text():
    wrapped = _wrap_completion_prompt(FakeTokenizer(), "summarize this")

    assert wrapped == {
        "encoder_prompt": {
            "prompt": "",
            "multi_modal_data": {"text": "summarize this"},
            "cache_salt": (
                "bart-enc:de6da4a29ce3cf585135c305c182ba1ee3ffb569"
                "34843469c04a4ab015287297"
            ),
        },
        "decoder_prompt": {"prompt_token_ids": [0]},
    }


def test_wrap_completion_prompt_keeps_encoder_decoder_prompt():
    prompt = {
        "encoder_prompt": {"prompt": "", "multi_modal_data": {"text": "x"}},
        "decoder_prompt": {"prompt_token_ids": [0]},
    }

    assert _wrap_completion_prompt(FakeTokenizer(), prompt) is prompt


def test_wrap_completion_prompt_decodes_token_ids():
    wrapped = _wrap_completion_prompt(
        FakeTokenizer(),
        {"prompt_token_ids": [1, 2, 3], "cache_salt": "existing"},
    )

    assert wrapped["encoder_prompt"]["multi_modal_data"] == {
        "text": "decoded:1,2,3"
    }
    assert wrapped["encoder_prompt"]["cache_salt"] == "existing"


def test_is_bart_family_model_matches_architecture_fallback():
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type=""),
        architectures=["Florence2ForConditionalGeneration"],
    )

    assert _is_bart_family_model(model_config)
