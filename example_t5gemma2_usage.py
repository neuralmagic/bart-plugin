#!/usr/bin/env python3
"""
Example usage of the vLLM BART plugin with T5Gemma2.

This script demonstrates how to use T5Gemma2 models with vLLM
after installing the BART plugin and the custom transformers fork.
"""
import vllm_bart_plugin
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset


def main():
    """Run T5Gemma2 model examples."""
    model_name = "google/t5gemma-2-270m-270m"

    print(f"Loading {model_name}...")
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=1024,
    )
    
    params = SamplingParams(
        temperature=0.0,
        max_tokens=64,
    )
    
    outputs = llm.generate(
        [
            {  # Simple text-to-text inference
                "prompt": "Translate English to French: The president of the United States is",
            },
            {  # Explicit encoder/decoder prompt
                "encoder_prompt": {
                    "prompt": "",
                    "multi_modal_data": {
                        "text": "Summarize: Machine learning is a field of study in artificial intelligence.",
                    },
                },
                "decoder_prompt": "Machine",
            },
            {  # Multimodal inference example (if the model supports vision tasks via its SigLIP encoder)
                "prompt": "Describe this image in detail.",
                "multi_modal_data": {"image": ImageAsset("stop_sign").pil_image},
            },
        ],
        sampling_params=params,
    )
    
    for i, o in enumerate(outputs):
        generated_text = o.outputs[0].text
        print(f"\n--- Output {i+1} ---")
        print(generated_text)


if __name__ == "__main__":
    main()
