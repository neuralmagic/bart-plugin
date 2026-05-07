"""Regression tests for vLLM 0.18 compatibility in the BART processor."""

import torch


def test_text_data_parser_handles_v018_empty_inputs():
    from vllm_bart_plugin.bart import TextDataParser

    parser = TextDataParser()

    assert parser._parse_text_data("") is None
    assert parser._parse_text_data([]) is None


def test_create_encoder_prompt_uses_placeholder_token():
    from vllm_bart_plugin.bart import BartMultiModalProcessor

    processor = BartMultiModalProcessor.__new__(BartMultiModalProcessor)

    assert processor.create_encoder_prompt("<s>decoder text", {"texts": ["encoder text"]}) == [0]


def test_call_hf_processor_accepts_pretokenized_decoder_prompt():
    from vllm_bart_plugin.bart import BartMultiModalProcessor

    class FakeTokenizer:
        def __call__(self, text, return_tensors="pt", **kwargs):
            if text == "encoder text":
                return {"input_ids": torch.tensor([[11, 12, 13]])}
            return {"input_ids": torch.tensor([[21, 22]])}

    class FakeInfo:
        def get_tokenizer(self):
            return FakeTokenizer()

    processor = BartMultiModalProcessor.__new__(BartMultiModalProcessor)
    processor.info = FakeInfo()

    out = processor._call_hf_processor(
        [7, 8, 9],
        {"texts": ["encoder text"]},
        {},
        {},
    )

    assert torch.equal(out["encoder_input_ids"], torch.tensor([[11, 12, 13]]))
    assert torch.equal(out["input_ids"], torch.tensor([[7, 8, 9]]))
