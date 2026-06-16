from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.kv_cache_interface import CrossAttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.states import RequestState
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class BartAttnMetadata(ModelSpecificAttnMetadata):
    encoder_seq_lens: dict[int, tuple[torch.Tensor, np.ndarray]]

    def get_extra_common_attn_kwargs(
        self,
        kv_cache_group_id: int,
        num_reqs: int,
    ) -> dict[str, Any]:
        encoder_seq_lens = self.encoder_seq_lens.get(kv_cache_group_id)
        if encoder_seq_lens is None:
            return {}
        encoder_seq_lens_gpu, encoder_seq_lens_cpu = encoder_seq_lens
        return {
            "encoder_seq_lens": encoder_seq_lens_gpu[:num_reqs],
            "encoder_seq_lens_cpu": encoder_seq_lens_cpu[:num_reqs],
        }


class BartEncoderDecoderModelState(DefaultModelState):
    """MRV2 state for BART-family encoder-decoder models."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.beam_sampler: Any | None = None
        self.encoder_outputs: list[torch.Tensor] = []
        self.encoder_seq_lens_gpu = torch.zeros(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        self.max_encoder_len = getattr(
            self.model_config.hf_config,
            "max_source_positions",
            self.max_model_len,
        )

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        if self.beam_sampler is None:
            return
        prompt_token_ids = (
            new_req_data.prefill_token_ids or new_req_data.prompt_token_ids
        )
        self.beam_sampler.register_request(
            new_req_data.req_id,
            new_req_data.sampling_params,
            prompt_token_ids,
        )

    def remove_request(self, req_id: str) -> None:
        if self.beam_sampler is not None:
            self.beam_sampler.remove_request(req_id)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        try:
            from vllm_beam_search.mrv2_sampler import BeamSearchMRV2Sampler
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("vllm_beam_search"):
                return None
            raise

        self.beam_sampler = BeamSearchMRV2Sampler(
            sampler,
            self.vllm_config,
            self.device,
        )
        block_tables = getattr(self, "_vllm_beam_block_tables", None)
        self_attn_groups = getattr(self, "_vllm_beam_self_attn_groups", ())
        if block_tables is not None:
            self.beam_sampler.set_block_tables(block_tables, self_attn_groups)
        return self.beam_sampler, None

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor,
    ) -> None:
        super().postprocess_state(idx_mapping, num_sampled)
        if self.beam_sampler is not None:
            self.beam_sampler.apply_pending_rewrites()

    def get_mm_embeddings(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
        input_batch: InputBatch,
    ) -> None:
        encoder_inputs = {
            req_id: req_encoder_inputs
            for req_id in input_batch.req_ids
            if (req_encoder_inputs := scheduled_encoder_inputs.get(req_id, []))
        }
        _, mm_kwargs = self.encoder_runner.prepare_mm_inputs(encoder_inputs)
        self.encoder_outputs = (
            self.encoder_runner.execute_mm_encoder(mm_kwargs) if mm_kwargs else []
        )

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        if not self.encoder_outputs:
            return {}
        encoder_outputs = self.encoder_outputs
        self.encoder_outputs = []
        return {"encoder_outputs": encoder_outputs}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {}

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        bart_attn_metadata = BartAttnMetadata(
            self._get_encoder_seq_lens(
                input_batch.req_ids, num_reqs, attn_groups, for_capture
            )
        )
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        max_seq_len = (
            self.max_model_len
            if for_capture
            else int(seq_lens_cpu_upper_bound[:num_reqs].max().item())
        )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            model_specific_attn_metadata=bart_attn_metadata,
            for_cudagraph_capture=for_capture,
        )

    def _get_encoder_seq_lens(
        self,
        req_ids: list[str],
        num_reqs: int,
        attn_groups: list[list[AttentionGroup]],
        for_capture: bool,
    ) -> dict[int, tuple[torch.Tensor, np.ndarray]]:
        encoder_seq_lens_np = np.zeros(num_reqs, dtype=np.int32)
        if for_capture:
            encoder_seq_lens_np[:] = self.max_encoder_len
        else:
            for req_index, req_id in enumerate(req_ids):
                mm_features = self.encoder_cache.mm_features.get(req_id, [])
                encoder_seq_lens_np[req_index] = sum(
                    feature.mm_position.length for feature in mm_features
                )

        self.encoder_seq_lens_gpu[:num_reqs].copy_(
            torch.from_numpy(encoder_seq_lens_np), non_blocking=True
        )
        encoder_seq_lens_gpu = self.encoder_seq_lens_gpu[:num_reqs]

        seq_lens_by_group: dict[int, tuple[torch.Tensor, np.ndarray]] = {}
        for kv_cache_group_idx, groups in enumerate(attn_groups):
            has_cross_attn = any(
                isinstance(attn_group.kv_cache_spec, CrossAttentionSpec)
                for attn_group in groups
            )
            if has_cross_attn:
                seq_lens_by_group[kv_cache_group_idx] = (
                    encoder_seq_lens_gpu,
                    encoder_seq_lens_np,
                )
        return seq_lens_by_group
