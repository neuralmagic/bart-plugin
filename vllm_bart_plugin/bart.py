# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Derived from BART implementation posted on HuggingFace; license below:
#
# coding=utf-8
# Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BART model."""

import math
import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import BartConfig
from transformers.utils import logging

from vllm.config import CacheConfig, VllmConfig
from vllm.config.lora import LoRAConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.cross_attention import CrossAttention
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    cast_overflow_tensors,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    ModalityData,
    ModalityDataItems,
    ModalityDataParser,
    MultiModalDataItems,
    MultiModalDataParser,
    ProcessorBatchItems,
)
from vllm.multimodal.processing import (
    BaseProcessingInfo,
    EncDecMultiModalProcessor,
    PromptUpdate,
)
from vllm.multimodal.processing.dummy_inputs import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.collection_utils import is_list_of
from vllm.v1.attention.backend import AttentionType

logger = logging.get_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean-ish environment variable.

    Accepted truthy: 1, true, yes, on
    Accepted falsy: 0, false, no, off
    """
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip().lower()
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    logger.warning("Unrecognized value for %s=%r; using default=%s", name, val, default)
    return default


def get_bsz_seq_len(input_ids):
    shp = input_ids.shape
    ndim = len(shp)
    if ndim == 1:
        return 1, input_ids.numel()
    else:
        return shp[:2]


class BartLearnedPositionalEmbedding(VocabParallelEmbedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is
        # specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately.
        # Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(
        self,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        return super().forward(positions + self.offset)


class BartScaledWordEmbedding(VocabParallelEmbedding):
    """
    This module overrides VocabParallelEmbedding's
    forward by multiplying with embeddings scale.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) * self.embed_scale


class BartParallelLMHead(ParallelLMHead):
    """
    This module overrides ParallelLMHead's
    forward by dividing by embeddings scale,
    yielding effectively the inverse of
    BartScaledWordEmbedding
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, embed_scale: float = 1.0
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.embed_scale = embed_scale

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().forward(input_ids) / self.embed_scale


class BartEncoderAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: BartConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = MMEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        is_2d = q.dim() == 2
        if is_2d:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            v = v.unsqueeze(0)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        if is_2d:
            output = output.squeeze(0)
        return output


class BartDecoderSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: BartConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.d_model,
            self.d_model // self.total_num_heads,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        attn_output = self.attn(q, k, v)

        output, _ = self.out_proj(attn_output)
        return output


class BartCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        config: BartConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.d_model = config.d_model
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = self.total_num_heads
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.kv_size = self.total_num_kv_heads * self.head_dim

        # Q_proj for projecting decoder hidden states
        self.q_proj = ColumnParallelLinear(
            input_size=embed_dim,
            output_size=embed_dim,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj",
        )

        # KV_proj for projecting encoder hidden states with no overhead of
        # unused Q_proj by setting total_num_heads to 0
        self.kv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=0,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_proj",
        )

        self.out_proj = RowParallelLinear(
            embed_dim,
            embed_dim,
            bias=bias,
            quant_config=quant_config,
        )

        tp_world_size = get_tensor_model_parallel_world_size()
        assert self.total_num_heads % tp_world_size == 0
        self.num_heads = self.total_num_heads // tp_world_size

        if self.total_num_kv_heads >= tp_world_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_world_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_world_size % self.total_num_kv_heads == 0
        self.num_kv_heads = self.num_heads  # No GQA in bart
        self.attn = CrossAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.ENCODER_DECODER,
        )

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Input shape: Batch x Time x Channel"""

        q, _ = self.q_proj(decoder_hidden_states)

        # Encoder hidden states are only computed once during prefill phase.
        # Afterwards, the keys and values should be available in the kv-cache.
        if encoder_hidden_states is not None:
            kv, _ = self.kv_proj(encoder_hidden_states)
            k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        else:
            k = v = None

        attn_output = self.attn(q, k, v)
        output, _ = self.out_proj(attn_output)
        return output


class BartEncoderLayer(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.activation_fn = get_act_fn(config.activation_function)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.act = get_act_fn("gelu")
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            hidden_states
                torch.Tensor of *encoder* input embeddings.
        Returns:
            Encoder layer output torch.Tensor
        """
        residual = hidden_states
        hidden_states = self.self_attn(hidden_states=hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16:
            hidden_states = cast_overflow_tensors(hidden_states)

        return hidden_states


class BartDecoderLayer(nn.Module):
    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartDecoderSelfAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            config=config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.activation_fn = get_act_fn(config.activation_function)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        """
        afeldman-nm: personally I would call this "cross-attention",
        however I left the name as "encoder_attn" to maintain consistency
        with the name of the pretrained weights.
        """
        self.encoder_attn = BartCrossAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            config=config,
            prefix=f"{prefix}.encoder_attn",
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        ffn_hidden_size = self.embed_dim
        ffn_intermediate_size = config.encoder_ffn_dim
        ffn_has_bias = True
        self.fc1 = ColumnParallelLinear(
            ffn_hidden_size,
            ffn_intermediate_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )
        self.fc2 = RowParallelLinear(
            ffn_intermediate_size,
            ffn_hidden_size,
            bias=ffn_has_bias,
            quant_config=quant_config,
        )

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_hidden_states
                torch.Tensor of *decoder* input embeddings.
            encoder_hidden_states
                torch.Tensor of *encoder* input embeddings.
        Returns:
            Decoder layer output torch.Tensor
        """
        residual = decoder_hidden_states

        # Self Attention
        hidden_states = self.self_attn(hidden_states=decoder_hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # Cross-Attention Block

        residual = hidden_states

        hidden_states = self.encoder_attn(
            decoder_hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        fc1_out, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(fc1_out)

        hidden_states, _ = self.fc2(hidden_states)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers*
    self attention layers. Each layer is a [`BartEncoderLayer`].
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()

        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        embed_dim = config.d_model
        self.max_source_positions = config.max_position_embeddings
        embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, embed_dim, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [
                BartEncoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.encoder_layers)
            ]
        )
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                Indices of *encoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            positions
                Positions of *encoder* input sequence tokens.
        Returns:
            Decoder output torch.Tensor
        """
        # retrieve input_ids and inputs_embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        embed_pos = self.embed_positions(positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states=hidden_states)
        return hidden_states


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers.
    Each layer is a [`BartDecoderLayer`]
    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(
        self,
        config: BartConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        lora_config: LoRAConfig | None = None,
        embed_tokens: nn.Embedding | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.lora_config = lora_config
        self.max_target_positions = config.max_position_embeddings
        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = BartScaledWordEmbedding(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )

        self.layers = nn.ModuleList(
            [
                BartDecoderLayer(
                    config,
                    cache_config,
                    quant_config,
                    prefix=f"{prefix}.layers.{layer_idx}",
                )
                for layer_idx in range(config.decoder_layers)
            ]
        )

        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_positions: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""
        Args:
            decoder_input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            decoder_positions
                Positions of *decoder* input sequence tokens.
            inputs_embeds:
                Tensor of decoder input embeddings
            encoder_hidden_states:
                Tensor of encoder output embeddings
        Returns:
            Decoder output torch.Tensor
        """
        if inputs_embeds is None:
            assert decoder_input_ids is not None
            inputs_embeds = self.embed_input_ids(decoder_input_ids)

        # embed positions
        embed_pos = self.embed_positions(decoder_positions)
        embed_pos = embed_pos.to(inputs_embeds.device)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)

        # decoder layers
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                decoder_hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
            )
        return hidden_states

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class BartModel(nn.Module, SupportsQuant):
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
    ]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config

        self.config = config

        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.encoder = BartEncoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.encoder"
        )
        self.decoder = BartDecoder(
            config, cache_config, quant_config=quant_config, prefix=f"{prefix}.decoder"
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        encoder_outputs: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                Indices of *decoder* input sequence tokens in the vocabulary.
                Padding will be ignored by default should you
                provide it.
            positions
                Positions of *decoder* input sequence tokens.
            encoder_input_ids
                Indices of *encoder* input sequence tokens in the vocabulary.
            encoder_positions:
                Positions of *encoder* input sequence tokens.
        Returns:
            Model output torch.Tensor
        """
        # decoder outputs consists of
        # (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids=input_ids,
            decoder_positions=positions,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_outputs,
        )

        return decoder_outputs

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]
        # Unify kv only for cross-attention, while keeping q separate
        cross_attn_stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("kv_proj", "k_proj", "k"),
            ("kv_proj", "v_proj", "v"),
        ]

        other_weights = []
        loaded_stacked_params = []
        model_params_dict = dict(self.named_parameters())

        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in cross_attn_stacked_params_mapping:
                if weight_name not in name or "encoder_attn" not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name not in model_params_dict:
                    continue
                param = model_params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                loaded_stacked_params.append(name)
                break
            else:
                for param_name, weight_name, shard_id in stacked_params_mapping:
                    if weight_name not in name or "encoder_attn" in name:
                        # Also skip q_proj in cross_attn which
                        # can be loaded normally
                        continue
                    name = name.replace(weight_name, param_name)
                    if name not in model_params_dict:
                        continue
                    param = model_params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_stacked_params.append(name)
                    break
                else:
                    if name in model_params_dict:
                        other_weights.append((name, loaded_weight))

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(other_weights)
        loaded_params.update(loaded_stacked_params)
        return loaded_params


class BartProcessingInfo(BaseProcessingInfo):
    """Processing information for BART encoder-decoder models."""

    def get_hf_config(self) -> BartConfig:
        return self.ctx.get_hf_config(BartConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # BART's encoder input is treated as a "text" modality
        # Like BART, mBART just has text for both encoder and decoder
        return {"text": 1}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # For BART, the encoder can handle up to max_position_embeddings tokens
        # Return this directly to avoid complex profiling
        config = self.get_hf_config()
        return {"text": config.max_position_embeddings}

    def get_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()

    def get_default_tok_params(self):
        return super().get_default_tok_params().with_kwargs(
            add_special_tokens=False
        )


class BartDummyInputsBuilder(BaseDummyInputsBuilder[BartProcessingInfo]):
    """Builds dummy inputs for profiling BART models."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        # For BART, the decoder prompt is separate from encoder
        # Return minimal dummy text for decoder
        return ""

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        # Return dummy encoder text for profiling
        num_texts = mm_counts.get("text", 0)
        if num_texts == 0:
            return {}

        # Create dummy encoder text of appropriate length
        # Use simple repeated words for profiling
        dummy_text = " ".join(["word"] * seq_len)
        return {"text": dummy_text}


# Allow "Text" as a Multimodal Modality for BART.
class TextProcessorItems(ProcessorBatchItems[str]):
    """
    Data items for text modality (BART encoder input is text).
    """

    def __init__(self, data) -> None:
        if data is None:
            data = [""]
        elif isinstance(data, str):
            data = [data]
        super().__init__(data, "text")


class TextDataParser(MultiModalDataParser):
    def __init__(self):
        super().__init__()

    def _parse_text_data(
        self,
        data: ModalityData[str],
    ) -> ModalityDataItems[Any, Any] | None:
        """Parse text data for BART."""
        if data is None or not len(data):
            return TextProcessorItems(None)

        # Text data should be a string or list of strings
        if isinstance(data, str) or is_list_of(data, str):
            return TextProcessorItems(data)
        else:
            raise TypeError(
                f"Text data must be a string or list of strings, got {type(data)}"
            )

    def _get_subparsers(self) -> Mapping[str, ModalityDataParser]:
        return {
            "text": self._parse_text_data,
        }


class BartMultiModalProcessor(EncDecMultiModalProcessor[BartProcessingInfo]):
    """Multimodal processor for BART encoder-decoder models."""

    def create_encoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        return [0]

    def create_decoder_prompt(
        self,
        prompt: str | list[int],
        mm_items: MultiModalDataItems,
    ) -> str | list[int]:
        # The decoder prompt is the original prompt
        return prompt

    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, object],
        tok_kwargs: Mapping[str, object],
    ):
        """
        BART doesn't have a HuggingFace Processor — it only has a tokenizer.

        Produces two sets of token IDs:
        - ``encoder_input_ids``: tokenized encoder text from ``mm_data["texts"]``
        - ``input_ids``: tokenized decoder prompt (used by the base class to
          build ``prompt_token_ids``)

        Encoder text is always tokenized with ``add_special_tokens=False`` to
        stay consistent with ``_get_prompt_updates``.
        """
        from transformers.feature_extraction_utils import BatchFeature

        tokenizer = self.info.get_tokenizer()

        # For BART encoder-decoder: check if we have encoder text data
        has_encoder_data = mm_data is not None and "texts" in mm_data
        result = {}

        # vLLM may pass add_special_tokens in tok_kwargs; we set it ourselves
        tok_kwargs = {k: v for k, v in tok_kwargs.items() if k != "add_special_tokens"}
        
        if has_encoder_data:
            encoder_texts = mm_data["texts"]
            encoder_text = encoder_texts[0] if encoder_texts else ""
            # Tokenize the encoder text from mm_data
            encoder_tokenized = tokenizer(
                encoder_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            result["encoder_input_ids"] = encoder_tokenized["input_ids"]

        # Always produce input_ids for the decoder prompt.
        if (
            isinstance(prompt, (list, tuple))
            and len(prompt) > 0
            and isinstance(prompt[0], int)
        ):
            result["input_ids"] = torch.tensor([prompt])
        else:
            prompt_tokenized = tokenizer(
                prompt if prompt else "",
                return_tensors="pt",
                **tok_kwargs,
            )
            result["input_ids"] = prompt_tokenized["input_ids"]

        return BatchFeature(result)

    def _get_mm_fields_config(
        self,
        hf_inputs,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        # For BART, encoder_input_ids from tokenization are the encoder input
        # and should be treated as multimodal kwargs for the "text" modality
        return dict(encoder_input_ids=MultiModalFieldConfig.batched("text"))

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Replace the single [0] encoder placeholder with N placeholder
        tokens, where N equals the tokenized length of the encoder text.

        The token count must use ``add_special_tokens=False`` to stay
        consistent with ``_call_hf_processor`` (which tokenizes the encoder
        text the same way).
        """
        from vllm.multimodal.processing import PromptReplacement

        # Get the number of text items to determine token count
        # For BART, we need to replace the placeholder [0] with the actual
        # number of encoder tokens from the text
        num_text_items = mm_items.get_count("text", strict=False)

        if num_text_items == 0:
            return []

        # Get the tokenized length - we'll use the input_ids from out_mm_kwargs
        # to determine how many tokens the text actually has
        text_items = mm_items.get_items("text", TextProcessorItems)
        tokenizer = self.info.get_tokenizer()

        # Tokenize the first text item to get the number of tokens
        text = text_items.get(0)
        num_tokens = len(tokenizer.encode(text, add_special_tokens=False))

        return [
            PromptReplacement(
                modality="text",
                target=[0],
                replacement=[0] * num_tokens,
            )
        ]

    def build_data_parser(self) -> MultiModalDataParser:
        return TextDataParser()


@MULTIMODAL_REGISTRY.register_processor(
    BartMultiModalProcessor,
    info=BartProcessingInfo,
    dummy_inputs=BartDummyInputsBuilder,
)
class BartForConditionalGeneration(nn.Module, SupportsQuant, SupportsMultiModal):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "decoder.": "model.decoder.",
            "encoder.": "model.encoder.",
            "shared.": "model.shared.",
        },
        orig_to_new_substr={
            "beta": "bias",
            "gamma": "weight",
            "LayerNorm": "layernorm",
        },
    )
    keys_to_ignore_on_load_missing = ["final_logits_bias"]

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config
        # currently all existing BART models have `tie_word_embeddings` enabled
        assert config.tie_word_embeddings
        self.config = config
        self.model = BartModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size

        embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        self.lm_head = BartParallelLMHead(
            config.vocab_size, config.d_model, embed_scale=embed_scale
        )
        # Bias added to logits after lm_head, matching HuggingFace approach
        self.register_buffer("final_logits_bias",
                             torch.zeros((1, config.vocab_size)))
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )
        # Optional optimization: run the encoder once over a padded batch of
        # encoder_input_ids (instead of N sequential encoder forwards).
        # Default is OFF to keep behavior stable unless explicitly enabled.
        self._encoder_max_seq_padding = _env_flag(
            "VLLM_BART_ENCODER_MAX_SEQ_PADDING", default=False
        )
        self._pad_id = getattr(self.config, "pad_token_id", None)
        if self._encoder_max_seq_padding and self._pad_id is None:
            logger.warning(
                "Pad token id is not set; disabling VLLM_BART_ENCODER_MAX_SEQ_PADDING."
            )
            self._encoder_max_seq_padding = False

    def get_language_model(self) -> nn.Module:
        return self.model.decoder

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.decoder.embed_tokens(input_ids)

    def embed_multimodal(self, **kwargs) -> MultiModalEmbeddings:
        encoder_input_ids_list = self._parse_and_validate_encoder_input(**kwargs)

        if not encoder_input_ids_list:
            raise ValueError(
                "encoder_input_ids_list is empty - this should not happen. "
                "Check that multimodal data is being passed correctly."
            )

        # Process each unique encoder input once and return a list of outputs.
        # Duplicate inputs (e.g. beams sharing a prompt) alias the same output
        # tensor, which is safe as long as consumers only read it.
        if not self._encoder_max_seq_padding:
            encoder_outputs: list[torch.Tensor] = []
            cached_outputs: dict[tuple[int, ...], torch.Tensor] = {}
            for encoder_input_ids in encoder_input_ids_list:
                cache_key = tuple(encoder_input_ids.reshape(-1).tolist())
                encoder_output = cached_outputs.get(cache_key)
                if encoder_output is not None:
                    encoder_outputs.append(encoder_output)
                    continue

                encoder_positions = torch.arange(
                    encoder_input_ids.size(-1),
                    dtype=torch.long,
                    device=encoder_input_ids.device,
                )

                encoder_output = self.model.encoder(
                    input_ids=encoder_input_ids.squeeze(0),
                    positions=encoder_positions,
                )
                cached_outputs[cache_key] = encoder_output
                encoder_outputs.append(encoder_output)
        else:
            # NOTE (NickLucche): Basic encoder batching optimization: BART input sequences
            # can have different lengths. Due to computational load of encoder being very
            # low here, we batch all sequences to run a single forward by max_seq padding.
            lengths = [t.numel() for t in encoder_input_ids_list]
            max_len = max(lengths) if lengths else 0
            assert max_len > 0, "Empty encoder_input_ids encountered."

            same_len = all(l == max_len for l in lengths)
            if len(encoder_input_ids_list) == 1:
                batch_encoder_input_ids = encoder_input_ids_list[0]
            elif same_len:
                # [1xD]xN =>NxD
                batch_encoder_input_ids = torch.cat(encoder_input_ids_list, dim=0)
            else:
                batch_encoder_input_ids = torch.full(
                    (len(encoder_input_ids_list), max_len),
                    fill_value=self._pad_id,
                    dtype=encoder_input_ids_list[0].dtype,
                    device=encoder_input_ids_list[0].device,
                )
                for i, t in enumerate(encoder_input_ids_list):
                    batch_encoder_input_ids[i, : t.numel()] = t.squeeze()

            # Create (B, T) positions: 0..T-1 for each item.
            batch_encoder_positions = (
                torch.arange(
                    max_len,
                    dtype=torch.long,
                    device=batch_encoder_input_ids.device,
                )
                .unsqueeze(0)
                .expand(batch_encoder_input_ids.size(0), -1)
            )

            # Run encoder once on the batch.
            batch_encoder_output = self.model.encoder(
                input_ids=batch_encoder_input_ids,
                positions=batch_encoder_positions,
            )

            # Split back into list[(T, H)] to match expected downstream format.
            # If we had to pad, slice back to the original lengths per item.
            encoder_outputs: list[torch.Tensor] = batch_encoder_output.unbind(dim=0)
            if not same_len:
                encoder_outputs = [
                    out[:l] for out, l in zip(encoder_outputs, lengths, strict=False)
                ]
        return encoder_outputs

    def _parse_and_validate_encoder_input(self, **kwargs: object) -> list[torch.Tensor]:
        encoder_input_ids = kwargs.get("encoder_input_ids", kwargs.get("input_ids"))

        if encoder_input_ids is None:
            return []

        if not isinstance(encoder_input_ids, (torch.Tensor, list)):
            raise ValueError(
                "Incorrect type of encoder input_ids. "
                f"Got type: {type(encoder_input_ids)}"
            )

        # Return as a list of tensors (one per item in the batch)
        if isinstance(encoder_input_ids, list):
            # Already a list - ensure each item is valid
            result = []
            for item in encoder_input_ids:
                if isinstance(item, torch.Tensor):
                    if item.dim() == 0:
                        item = item.unsqueeze(0)
                    result.append(item)
                else:
                    result.append(item)
            return result
        else:
            # [1xD]xN times
            return encoder_input_ids.unsqueeze(1).unbind(dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        encoder_outputs: list[torch.Tensor] | None = None,
        # num_encoder_outputs: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        r"""
        Args:
            input_ids
                torch.Tensor of *decoder* input token ids.
            positions
                torch.Tensor of *decoder* position indices.
        Keyword Args:
            encoder_input_ids (optional)
                torch.Tensor of *encoder* input token ids.
            encoder_positions (optional)
                torch.Tensor of *encoder* position indices
        Returns:
            Output torch.Tensor
        """
        # EncoderDecoderModelState passes an empty list on decode steps.
        enc_states = (
            torch.cat(encoder_outputs, dim=0) if encoder_outputs else None
        )

        return self.model(
            input_ids, positions, inputs_embeds, encoder_outputs=enc_states
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None:
            logits = logits + self.final_logits_bias
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights_tuple_list = list(weights)

        shared_embedding_weight = None
        # Load separately as buffers are not registered as nn.Module
        final_logits_bias_weight = None
        for name, loaded_weight in weights_tuple_list:
            if "final_logits_bias" in name:
                final_logits_bias_weight = loaded_weight
            elif (
                "shared.weight" in name
                or "encoder.embed_tokens.weight" in name
                or "decoder.embed_tokens.weight" in name
                or "lm_head.weight" in name
            ):
                if shared_embedding_weight is not None:
                    logger.warning(
                        "Shared weight embedding already loaded with name "
                        "%s, skipping. This is expected on facebook/bart-large"
                        " like models, where the same shared embedding is "
                        "present multiple times.", name)
                    continue
                shared_embedding_weight = loaded_weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["cls.", "pooler."]),
            skip_substrs=(["final_logits_bias"]),
        )
        loaded_params = loader.load_weights(
            weights_tuple_list, mapper=self.hf_to_vllm_mapper
        )

        # Load final_logits_bias if present in checkpoint
        if final_logits_bias_weight is not None:
            self.final_logits_bias.copy_(final_logits_bias_weight)
            loaded_params.add("final_logits_bias")

        if shared_embedding_weight is not None:
            weight_loader = getattr(
                self.lm_head.weight, "weight_loader", default_weight_loader
            )
            weight_loader(self.lm_head.weight, shared_embedding_weight)

            self.model.encoder.embed_tokens.weight = self.lm_head.weight
            self.model.decoder.embed_tokens.weight = self.lm_head.weight
            loaded_params.update(
                {
                    "model.encoder.embed_tokens.weight",
                    "lm_head.weight",
                    "model.decoder.embed_tokens.weight",
                }
            )

        # Add keys_to_ignore_on_load_missing to loaded_params so vLLM
        # doesn't report them as missing (they use default initialized values)
        for key in self.keys_to_ignore_on_load_missing:
            loaded_params.add(key)

        return loaded_params