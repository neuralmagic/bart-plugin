from __future__ import annotations

import contextvars
import hashlib
import os
import sys
import types

_PATCHES_INSTALLED = False
_ENCODER_TRACE = bool(int(os.getenv("VLLM_BART_ENCODER_TRACE", "0")))
_INPUT_TRACE = bool(int(os.getenv("VLLM_BART_INPUT_TRACE", "0")))
_KEEP_ENCODER_OVERFLOW_CHECK = (
    os.getenv("VLLM_BART_KEEP_ENCODER_OVERFLOW_CHECK", "0") == "1"
)
_BATCH_INVARIANT_LINEAR = os.getenv(
    "VLLM_BART_BATCH_INVARIANT_LINEAR",
    "0",
).lower()
_LINEAR_DIAG_CONTEXT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "bart_batch_invariant_linear_context",
    default=None,
)
_LINEAR_LAYER_SCOPES = (
    "encoder_attn_layer_",
    "encoder_ffn_layer_",
    "decoder_self_attn_layer_",
    "decoder_cross_attn_layer_",
    "decoder_ffn_layer_",
    "decoder_cross_ffn_layer_",
    "decoder_layer_",
)
_LINEAR_PROJECTION_LAYER_SCOPES = (
    "decoder_cross_attn_q_layer_",
    "decoder_cross_attn_kv_layer_",
    "decoder_cross_attn_out_layer_",
    "decoder_cross_attn_qkv_layer_",
    "decoder_cross_attn_qout_layer_",
    "decoder_cross_attn_kvout_layer_",
)


def _maybe_trace_encoder(message: str) -> None:
    if not _ENCODER_TRACE:
        return
    print(f"[BART_ENCODER_TRACE] {message}", file=sys.stderr, flush=True)


def _linear_diag_layer_index(scope_prefix: str) -> int | None:
    if not _BATCH_INVARIANT_LINEAR.startswith(scope_prefix):
        return None
    suffix = _BATCH_INVARIANT_LINEAR.removeprefix(scope_prefix)
    if not suffix.isdigit():
        return None
    return int(suffix)


def _safe_block_lengths_from_scheduler(scheduler, request_id: str) -> list[object]:
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if kv_cache_manager is None:
        return []
    try:
        return [len(group) for group in kv_cache_manager.get_block_ids(request_id)]
    except Exception as exc:  # pragma: no cover - trace-only fallback
        return [f"error:{type(exc).__name__}"]


def _beam_trace_context(scheduler, request_id: str) -> tuple[str | None, bool]:
    orig_id = None
    active_beam = False
    beam_to_group = getattr(scheduler, "beam_to_group", None)
    beam_groups = getattr(scheduler, "beam_groups", None)
    if isinstance(beam_to_group, dict):
        orig_id = beam_to_group.get(request_id)
    if orig_id is not None and isinstance(beam_groups, dict):
        group = beam_groups.get(orig_id)
        active_beams = getattr(group, "active_beams", None)
        if active_beams is not None:
            active_beam = request_id in active_beams
        else:
            beam_ids = getattr(group, "beam_request_ids", None)
            active_beam = beam_ids is not None and request_id in beam_ids
    return orig_id, active_beam


def _is_bart_family_model(model_config) -> bool:
    hf_config = getattr(model_config, "hf_config", None)
    model_type = str(getattr(hf_config, "model_type", "") or "").lower()
    architectures = getattr(model_config, "architectures", None)
    if not architectures and hf_config is not None:
        architectures = getattr(hf_config, "architectures", None)
    arch_text = " ".join(str(arch) for arch in (architectures or ()))
    return (
        model_type in {"bart", "mbart", "florence2"}
        or "Bart" in arch_text
        or "Florence2" in arch_text
    )


def _encoder_cache_salt(
    prompt_text: str, existing_salt: str | None = None
) -> str | None:
    if existing_salt:
        return existing_salt
    if not prompt_text:
        return None
    digest = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    return f"bart-enc:{digest}"


def _patch_openai_serving() -> None:
    def _decoder_bos_prompt(tokenizer):
        bos = getattr(tokenizer, "bos_token_id", None)
        if bos is None:
            return None
        return {"prompt_token_ids": [int(bos)]}

    def _wrap_completion_prompt(tokenizer, engine_prompt):
        if isinstance(engine_prompt, dict) and "encoder_prompt" in engine_prompt:
            return engine_prompt

        prompt_text: str | None
        cache_salt: str | None = None
        mm_processor_kwargs = None

        if isinstance(engine_prompt, str):
            prompt_text = engine_prompt
        elif isinstance(engine_prompt, list):
            if tokenizer is None:
                return engine_prompt
            prompt_text = tokenizer.decode(engine_prompt)
        else:
            prompt_text = engine_prompt.get("prompt")
            cache_salt = engine_prompt.get("cache_salt")
            mm_processor_kwargs = engine_prompt.get("mm_processor_kwargs")
            if prompt_text is None:
                prompt_token_ids = engine_prompt.get("prompt_token_ids")
                if prompt_token_ids is not None and tokenizer is not None:
                    prompt_text = tokenizer.decode(prompt_token_ids)

        decoder_prompt = _decoder_bos_prompt(tokenizer)
        if decoder_prompt is None:
            return engine_prompt

        if isinstance(engine_prompt, dict) and engine_prompt.get(
            "multi_modal_data"
        ) is not None:
            wrapped_prompt = dict(engine_prompt)
        elif prompt_text is None:
            return engine_prompt
        else:
            wrapped_prompt = {
                "prompt": "",
                "multi_modal_data": {"text": prompt_text},
            }
        if mm_processor_kwargs is not None:
            wrapped_prompt["mm_processor_kwargs"] = mm_processor_kwargs
        if cache_salt := _encoder_cache_salt(prompt_text, cache_salt):
            wrapped_prompt["cache_salt"] = cache_salt
        return {
            "encoder_prompt": wrapped_prompt,
            "decoder_prompt": decoder_prompt,
        }

    try:
        from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    except ImportError:
        OpenAIServingRender = None

    if (
        OpenAIServingRender is not None
        and not getattr(OpenAIServingRender, "_vllm_bart_prompt_patched", False)
    ):
        original_preprocess_completion = OpenAIServingRender.preprocess_completion

        def _wrap_prompt_input(self, prompt_input):
            tokenizer = self.renderer.tokenizer
            if prompt_input is None:
                return None
            if isinstance(prompt_input, str):
                return _wrap_completion_prompt(tokenizer, prompt_input)
            if isinstance(prompt_input, dict):
                return _wrap_completion_prompt(tokenizer, prompt_input)
            if isinstance(prompt_input, list):
                if not prompt_input:
                    return prompt_input
                if all(isinstance(t, int) for t in prompt_input):
                    return _wrap_completion_prompt(tokenizer, prompt_input)
                return [_wrap_completion_prompt(tokenizer, p) for p in prompt_input]
            return prompt_input

        async def patched_preprocess_completion(
            self,
            request,
            prompt_input,
            prompt_embeds,
            *,
            skip_mm_cache: bool = False,
        ):
            if (
                prompt_embeds is None
                and getattr(self.model_config, "is_encoder_decoder", False)
                and _is_bart_family_model(self.model_config)
            ):
                prompt_input = _wrap_prompt_input(self, prompt_input)
            return await original_preprocess_completion(
                self,
                request,
                prompt_input,
                prompt_embeds,
                skip_mm_cache=skip_mm_cache,
            )

        OpenAIServingRender.preprocess_completion = patched_preprocess_completion
        OpenAIServingRender._vllm_bart_prompt_patched = True

    try:
        from vllm.entrypoints.openai.serving_engine import OpenAIServing
    except ImportError:
        return

    if getattr(OpenAIServing, "_vllm_bart_prompt_patched", False):
        return

    original_process_inputs = OpenAIServing._process_inputs

    async def patched_process_inputs(
        self,
        request_id: str,
        engine_prompt,
        params,
        *,
        lora_request,
        trace_headers,
        priority: int,
    ):
        if (
            getattr(self.model_config, "is_encoder_decoder", False)
            and _is_bart_family_model(self.model_config)
        ):
            if not (
                isinstance(engine_prompt, dict)
                and "encoder_prompt" in engine_prompt
            ):
                engine_prompt = _wrap_completion_prompt(
                    self.input_processor.tokenizer, engine_prompt
                )
        return await original_process_inputs(
            self,
            request_id,
            engine_prompt,
            params,
            lora_request=lora_request,
            trace_headers=trace_headers,
            priority=priority,
        )

    OpenAIServing._process_inputs = patched_process_inputs
    OpenAIServing._vllm_bart_prompt_patched = True


def _patch_bart_multimodal_processing() -> None:
    """Force encoder text tokenization to add_special_tokens=False.

    The cached BartMultiModalProcessor._call_hf_processor tokenizes the
    encoder text with `**tok_kwargs`, whose `add_special_tokens=False`
    line is commented out — so the encoder input gets wrapped in
    <s>...</s>. Production/Triton/HF use the *unwrapped* tokenization;
    the extra special tokens shift the encoder hidden states and corrupt
    decoder logits (verified: with-specials top token after </s><s> is
    " Check", without-specials it is " Clock", matching Triton). Force
    the kwarg here so the encoder sees the same tokens as production.
    """
    try:
        from vllm_bart_plugin.bart import BartMultiModalProcessor
    except Exception:
        return

    if getattr(BartMultiModalProcessor, "_vllm_bart_enc_specials_patched", False):
        return

    original_call_hf = BartMultiModalProcessor._call_hf_processor

    def patched_call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        tok_kwargs = dict(tok_kwargs or {})
        tok_kwargs["add_special_tokens"] = False
        return original_call_hf(self, prompt, mm_data, mm_kwargs, tok_kwargs)

    BartMultiModalProcessor._call_hf_processor = patched_call_hf_processor
    BartMultiModalProcessor._vllm_bart_enc_specials_patched = True


def _patch_register_beam_search_logits_processor() -> None:
    """Register the BeamSearchLogitsProcessor for V1.

    This processor reads `_beam_group_id` / `_beam_index` from each
    request's `sampling_params.extra_args` (set by `BeamSearchScheduler`
    on the per-beam child Request objects) and drives the per-step
    beam-search logic from the hidden decoder BOS prompt state. BOS remains
    an ordinary candidate after that prompt, matching V0.

    Idempotent.
    """
    from vllm.v1.sample.logits_processor import BUILTIN_LOGITS_PROCESSORS
    from vllm_beam_search.logits_processor import BeamSearchLogitsProcessor

    if any(
        getattr(cls, "_is_beam_search_logitsproc", False)
        for cls in BUILTIN_LOGITS_PROCESSORS
    ):
        return
    BUILTIN_LOGITS_PROCESSORS.append(BeamSearchLogitsProcessor)


def _patch_cross_attention_manager() -> None:
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    if getattr(CrossAttentionManager, "_vllm_bart_plugin_patched", False):
        return

    def cache_blocks(self, request, num_tokens, *args, **kwargs) -> None:
        return None

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        **_ignored_kwargs,
    ):
        return tuple([] for _ in range(len(kv_cache_group_ids)))

    CrossAttentionManager.cache_blocks = cache_blocks
    CrossAttentionManager.find_longest_cache_hit = find_longest_cache_hit
    CrossAttentionManager._vllm_bart_plugin_patched = True


def _patch_hybrid_coordinator() -> None:
    return None


def _patch_cross_attention_backend() -> None:
    return None


def _patch_beam_scheduler() -> None:
    return None


def _patch_scheduler_encoder_handling() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_vllm_bart_plugin_encoder_patched", False):
        return

    original_try_schedule_encoder_inputs = Scheduler._try_schedule_encoder_inputs
    original_free_encoder_inputs = Scheduler._free_encoder_inputs

    def _num_output_tokens(request) -> int:
        return int(
            getattr(
                request,
                "num_output_tokens",
                len(getattr(request, "output_token_ids", ()) or ()),
            )
        )

    def patched_try_schedule_encoder_inputs(
        self,
        request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ):
        trace_request = (
            _ENCODER_TRACE
            and self.is_encoder_decoder
            and ":beam:" in getattr(request, "request_id", "")
        )
        original_num_computed_tokens = num_computed_tokens
        original_num_new_tokens = num_new_tokens
        if self.is_encoder_decoder and _num_output_tokens(request) == 0:
            if not getattr(request, "_beam_cross_attn_seeded", False):
                num_computed_tokens = 0
        result = original_try_schedule_encoder_inputs(
            self,
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            shift_computed_tokens=shift_computed_tokens,
        )
        if trace_request:
            encoder_inputs_to_schedule, adjusted_num_new_tokens, _, external_load = (
                result
            )
            _maybe_trace_encoder(
                "scheduler "
                f"req_id={request.request_id} "
                f"seeded={getattr(request, '_beam_cross_attn_seeded', False)} "
                f"output_tokens={_num_output_tokens(request)} "
                f"orig_num_computed_tokens={original_num_computed_tokens} "
                f"effective_num_computed_tokens={num_computed_tokens} "
                f"orig_num_new_tokens={original_num_new_tokens} "
                f"adjusted_num_new_tokens={adjusted_num_new_tokens} "
                f"scheduled_inputs={list(encoder_inputs_to_schedule)} "
                f"external_inputs={list(external_load)}"
            )
        return result

    def patched_free_encoder_inputs(self, request) -> None:
        if self.is_encoder_decoder and _num_output_tokens(request) == 0:
            return
        return original_free_encoder_inputs(self, request)

    Scheduler._try_schedule_encoder_inputs = patched_try_schedule_encoder_inputs
    Scheduler._free_encoder_inputs = patched_free_encoder_inputs
    Scheduler._vllm_bart_plugin_encoder_patched = True


def _patch_gpu_model_runner_encoder_tracing() -> None:
    if not _ENCODER_TRACE:
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_bart_encoder_trace_patched", False):
        return

    original_execute_mm_encoder = GPUModelRunner._execute_mm_encoder

    def patched_execute_mm_encoder(self, scheduler_output):
        if _ENCODER_TRACE and self.model_config.is_encoder_decoder:
            scheduled_encoder_inputs = scheduler_output.scheduled_encoder_inputs
            if scheduled_encoder_inputs:
                parts: list[str] = []
                for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
                    req_state = self.requests.get(req_id)
                    if req_state is None:
                        continue
                    parts.append(
                        f"{req_id}|inputs={list(encoder_input_ids)}|"
                        f"computed={req_state.num_computed_tokens}|"
                        f"prompt={len(req_state.prompt_token_ids)}|"
                        f"output={len(req_state.output_token_ids)}"
                    )
                if parts:
                    _maybe_trace_encoder(
                        "worker execute_mm_encoder " + " ; ".join(parts)
                    )
        return original_execute_mm_encoder(self, scheduler_output)

    GPUModelRunner._execute_mm_encoder = patched_execute_mm_encoder
    GPUModelRunner._bart_encoder_trace_patched = True


def _patch_gpu_model_runner_encoder_dedup() -> None:
    """Reuse identical BART encoder outputs within a worker step.

    Current V1 does not use the encoder cache for encoder-decoder models, so
    beam children with the same encoder prompt can arrive as separate encoder
    work items. Computing one output per unique mm hash is faithful: identical
    encoder inputs produce identical cross-attention memory, and downstream
    decoder code only reads these tensors.
    """
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_bart_encoder_dedup_patched", False):
        return

    original_execute_mm_encoder = GPUModelRunner._execute_mm_encoder

    def patched_execute_mm_encoder(self, scheduler_output):
        if (
            not getattr(self.model_config, "is_encoder_decoder", False)
            or not _is_bart_family_model(self.model_config)
            or getattr(self, "lora_config", None)
        ):
            return original_execute_mm_encoder(self, scheduler_output)

        (
            mm_hashes,
            mm_kwargs,
            mm_lora_refs,
        ) = self._batch_mm_inputs_from_scheduler(scheduler_output)
        if len(mm_hashes) <= 1:
            return original_execute_mm_encoder(self, scheduler_output)

        output_by_hash = {
            mm_hash: self.encoder_cache[mm_hash]
            for mm_hash in dict.fromkeys(mm_hashes)
            if mm_hash in self.encoder_cache
        }
        unique_hashes: list[str] = []
        unique_kwargs = []
        unique_lora_refs = []
        seen = set(output_by_hash)
        for mm_hash, kwargs, lora_ref in zip(mm_hashes, mm_kwargs, mm_lora_refs):
            if mm_hash in seen:
                continue
            seen.add(mm_hash)
            unique_hashes.append(mm_hash)
            unique_kwargs.append(kwargs)
            unique_lora_refs.append(lora_ref)

        if len(unique_hashes) == len(mm_hashes):
            return original_execute_mm_encoder(self, scheduler_output)

        if unique_hashes:
            original_batch = self._batch_mm_inputs_from_scheduler

            def unique_batch(_self, _scheduler_output):
                return unique_hashes, unique_kwargs, unique_lora_refs

            self._batch_mm_inputs_from_scheduler = types.MethodType(unique_batch, self)
            try:
                unique_outputs = original_execute_mm_encoder(self, scheduler_output)
            finally:
                self._batch_mm_inputs_from_scheduler = original_batch

            output_by_hash.update(zip(unique_hashes, unique_outputs))

        if _ENCODER_TRACE:
            _maybe_trace_encoder(
                "worker encoder_dedup "
                f"items={len(mm_hashes)} unique={len(set(mm_hashes))} "
                f"computed={len(unique_hashes)}"
            )

        return [output_by_hash[mm_hash] for mm_hash in mm_hashes]

    GPUModelRunner._execute_mm_encoder = patched_execute_mm_encoder
    GPUModelRunner._bart_encoder_dedup_patched = True


def _patch_gpu_model_runner_input_tracing() -> None:
    if not _INPUT_TRACE:
        return

    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if getattr(GPUModelRunner, "_bart_input_trace_patched", False):
        return

    original_prepare_inputs = GPUModelRunner._prepare_inputs

    def patched_prepare_inputs(self, scheduler_output, num_scheduled_tokens):
        result = original_prepare_inputs(self, scheduler_output, num_scheduled_tokens)

        parts: list[str] = []
        offset = 0
        for req_idx, req_id in enumerate(self.input_batch.req_ids):
            if ":beam:" not in req_id:
                offset += int(num_scheduled_tokens[req_idx])
                continue
            count = int(num_scheduled_tokens[req_idx])
            req_state = self.requests.get(req_id)
            if req_state is None:
                offset += count
                continue
            toks = self.input_ids.cpu[offset : offset + count].tolist()
            parts.append(
                f"{req_id}|computed={req_state.num_computed_tokens}|"
                f"prompt={len(req_state.prompt_token_ids)}|"
                f"output={len(req_state.output_token_ids)}|"
                f"scheduled={count}|input={toks}|"
                f"out={list(req_state.output_token_ids)}"
            )
            offset += count
        if parts:
            print(
                "[BART_INPUT_TRACE] " + " ; ".join(parts),
                file=sys.stderr,
                flush=True,
            )

        return result

    GPUModelRunner._prepare_inputs = patched_prepare_inputs
    GPUModelRunner._bart_input_trace_patched = True


def _patch_scheduler_free_tracing() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler, "_bart_free_trace_patched", False):
        return

    original_finish_requests = Scheduler.finish_requests
    original_free_request = Scheduler._free_request

    def patched_finish_requests(self, request_ids, finished_status):
        if isinstance(request_ids, str):
            req_ids = (request_ids,)
        elif request_ids is None:
            req_ids = tuple(self.requests.keys())
        else:
            req_ids = tuple(request_ids)

        if _ENCODER_TRACE:
            for req_id in req_ids:
                orig_id, active_beam = _beam_trace_context(self, req_id)
                if ":beam:" not in req_id and orig_id is None:
                    continue
                request = self.requests.get(req_id)
                _maybe_trace_encoder(
                    "finish_requests_start "
                    f"req_id={req_id} "
                    f"finished_status={finished_status.name} "
                    f"req_status={getattr(request, 'status', None)} "
                    f"orig_group={orig_id} "
                    f"active_beam={active_beam} "
                    f"num_computed_tokens={getattr(request, 'num_computed_tokens', None)} "
                    f"block_lengths={_safe_block_lengths_from_scheduler(self, req_id)}"
                )

        result = original_finish_requests(self, request_ids, finished_status)

        if _ENCODER_TRACE:
            for req_id in req_ids:
                orig_id, active_beam = _beam_trace_context(self, req_id)
                if ":beam:" not in req_id and orig_id is None:
                    continue
                request = self.requests.get(req_id)
                _maybe_trace_encoder(
                    "finish_requests_done "
                    f"req_id={req_id} "
                    f"request_exists={request is not None} "
                    f"orig_group={orig_id} "
                    f"active_beam={active_beam} "
                    f"finished_recorded={req_id in self.finished_req_ids} "
                    f"block_lengths={_safe_block_lengths_from_scheduler(self, req_id)}"
                )
        return result

    def patched_free_request(self, request, delay_free_blocks=False, **_ignored_kwargs):
        req_id = request.request_id
        orig_id, active_beam = _beam_trace_context(self, req_id)
        trace_request = _ENCODER_TRACE and (":beam:" in req_id or orig_id is not None)
        if trace_request:
            _maybe_trace_encoder(
                "free_request_start "
                f"req_id={req_id} "
                f"req_status={request.status.name} "
                f"orig_group={orig_id} "
                f"active_beam={active_beam} "
                f"num_computed_tokens={request.num_computed_tokens} "
                f"block_lengths={_safe_block_lengths_from_scheduler(self, req_id)}"
            )
        result = original_free_request(
            self, request, delay_free_blocks=delay_free_blocks
        )
        if trace_request:
            orig_id_after, active_beam_after = _beam_trace_context(self, req_id)
            _maybe_trace_encoder(
                "free_request_done "
                f"req_id={req_id} "
                f"request_exists={req_id in self.requests} "
                f"orig_group={orig_id_after} "
                f"active_beam={active_beam_after} "
                f"finished_recorded={req_id in self.finished_req_ids} "
                f"block_lengths={_safe_block_lengths_from_scheduler(self, req_id)}"
            )
        return result

    Scheduler.finish_requests = patched_finish_requests
    Scheduler._free_request = patched_free_request
    Scheduler._bart_free_trace_patched = True


def _patch_kv_free_tracing() -> None:
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    if getattr(KVCacheManager, "_bart_free_trace_patched", False):
        return

    original_free = KVCacheManager.free

    def patched_free(self, request):
        req_id = request.request_id
        trace_request = _ENCODER_TRACE and ":beam:" in req_id
        if trace_request:
            before = self.get_block_ids(req_id)
            _maybe_trace_encoder(
                "kv_free_start "
                f"req_id={req_id} "
                f"req_status={request.status.name} "
                f"num_computed_tokens={request.num_computed_tokens} "
                f"block_lengths={[len(group) for group in before]}"
            )
        result = original_free(self, request)
        if trace_request:
            after = self.get_block_ids(req_id)
            _maybe_trace_encoder(
                "kv_free_done "
                f"req_id={req_id} "
                f"block_lengths={[len(group) for group in after]}"
            )
        return result

    KVCacheManager.free = patched_free
    KVCacheManager._bart_free_trace_patched = True


def _patch_kv_allocate_encoder_tokens() -> None:
    """Cap cross-attention block allocation and fast-path seeded beam children."""
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.single_type_kv_cache_manager import CrossAttentionManager

    if getattr(KVCacheManager, "_bart_allocate_encoder_patched", False):
        return

    original_allocate_slots = KVCacheManager.allocate_slots

    def patched_allocate_slots(
        self,
        request,
        num_new_tokens,
        num_new_computed_tokens=0,
        new_computed_blocks=None,
        num_lookahead_tokens=0,
        num_external_computed_tokens=0,
        delay_cache_blocks=False,
        num_encoder_tokens=0,
        full_sequence_must_fit=False,
        reserved_blocks=0,
        has_scheduled_reqs=True,
        **_ignored_kwargs,
    ):
        if num_encoder_tokens > 0:
            if getattr(request, "_beam_cross_attn_seeded", False):
                num_encoder_tokens = 0
            else:
                mm_features = getattr(request, "mm_features", None)
                if mm_features:
                    actual = sum(f.mm_position.length for f in mm_features)
                    if actual > 0:
                        num_encoder_tokens = actual

        fast_beam_path = (
            getattr(request, "_beam_cross_attn_seeded", False)
            and request.num_computed_tokens > 0
            and num_new_computed_tokens == 0
            and num_lookahead_tokens == 0
            and not delay_cache_blocks
            and num_encoder_tokens == 0
            and (new_computed_blocks is None or not any(new_computed_blocks.blocks))
        )
        if fast_beam_path:
            self.coordinator.remove_skipped_blocks(
                request.request_id, request.num_computed_tokens
            )
            num_tokens_need_slot = min(
                request.num_computed_tokens + num_new_tokens,
                self.max_model_len,
            )
            num_blocks_to_allocate = 0
            for manager in self.coordinator.single_type_managers:
                if isinstance(manager, CrossAttentionManager):
                    continue
                if manager.block_size != 1:
                    fast_beam_path = False
                    break
                req_blocks = manager.req_to_blocks.get(request.request_id)
                if req_blocks is None:
                    fast_beam_path = False
                    break
                num_blocks_to_allocate += max(
                    num_tokens_need_slot - len(req_blocks), 0
                )
            if fast_beam_path:
                if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
                    return None
                new_blocks = self.coordinator.allocate_new_blocks(
                    request.request_id, num_tokens_need_slot, 0
                )
                if request.block_hashes:
                    num_tokens_to_cache = min(
                        request.num_computed_tokens + num_new_tokens,
                        request.num_tokens,
                    )
                    self.coordinator.cache_blocks(request, num_tokens_to_cache)
                return self.create_kv_cache_blocks(new_blocks)

        return original_allocate_slots(
            self=self,
            request=request,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )

    KVCacheManager.allocate_slots = patched_allocate_slots
    KVCacheManager._bart_allocate_encoder_patched = True


def _patch_encoder_overflow_check() -> None:
    """Remove the expensive isinf/isnan overflow check from BartEncoderLayer.
    Each check forces a GPU->CPU sync via .any(), and with 12 layers * N
    requests per batch, this creates hundreds of sync points per step."""
    if _KEEP_ENCODER_OVERFLOW_CHECK:
        return

    try:
        from vllm_bart_plugin.bart import BartEncoderLayer
    except ImportError:
        return

    if getattr(BartEncoderLayer, "_overflow_check_patched", False):
        return

    def _forward_no_overflow(self, hidden_states):
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

        return hidden_states

    BartEncoderLayer.forward = _forward_no_overflow
    BartEncoderLayer._overflow_check_patched = True


def _patch_batch_invariant_linear_diagnostic() -> None:
    """Opt-in numeric diagnostic for BART unquantized linear/LM-head GEMMs."""
    if _BATCH_INVARIANT_LINEAR in {"0", "false", "off", "no"}:
        return
    layer_scoped_mode = any(
        _BATCH_INVARIANT_LINEAR.startswith(prefix)
        and _linear_diag_layer_index(prefix) is not None
        for prefix in _LINEAR_LAYER_SCOPES
    )
    projection_scoped_mode = any(
        _BATCH_INVARIANT_LINEAR.startswith(prefix)
        and _linear_diag_layer_index(prefix) is not None
        for prefix in _LINEAR_PROJECTION_LAYER_SCOPES
    )
    full_model_modes = {
        "1",
        "true",
        "on",
        "yes",
        "all",
        "model",
    }
    scoped_model_modes = {
        "encoder",
        "decoder",
        "encoder_attn",
        "encoder_ffn",
        "decoder_self_attn",
        "decoder_cross_attn",
        "decoder_ffn",
    }
    patch_model_linears = _BATCH_INVARIANT_LINEAR in {
        *full_model_modes,
        *scoped_model_modes,
    } or layer_scoped_mode or projection_scoped_mode
    patch_lm_head = _BATCH_INVARIANT_LINEAR in {
        "1",
        "true",
        "on",
        "yes",
        "all",
        "lm_head",
        "head",
    }
    if not (patch_model_linears or patch_lm_head):
        raise ValueError(
            "VLLM_BART_BATCH_INVARIANT_LINEAR must be one of "
            "0, all, model, encoder, decoder, encoder_attn, encoder_ffn, "
            "decoder_self_attn, decoder_cross_attn, decoder_ffn, "
            "<scope>_layer_<idx>, decoder_cross_ffn_layer_<idx>, "
            "decoder_layer_<idx>, "
            "decoder_cross_attn_<q|kv|out|qkv|qout|kvout>_layer_<idx>, "
            "or lm_head"
        )

    from vllm.model_executor.layers.batch_invariant import linear_batch_invariant
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        UnquantizedEmbeddingMethod,
    )
    from vllm.platforms import current_platform

    if not current_platform.is_cuda_alike():
        return

    def should_patch_model_linear() -> bool:
        if _BATCH_INVARIANT_LINEAR in full_model_modes:
            return True
        return _LINEAR_DIAG_CONTEXT.get() == _BATCH_INVARIANT_LINEAR

    if patch_model_linears and not getattr(
        UnquantizedLinearMethod,
        "_bart_batch_invariant_linear_patched",
        False,
    ):
        original_linear_apply = UnquantizedLinearMethod.apply

        def linear_apply(self, layer, x, bias=None):
            if should_patch_model_linear():
                return linear_batch_invariant(x, layer.weight, bias)
            return original_linear_apply(self, layer, x, bias)

        UnquantizedLinearMethod.apply = linear_apply
        UnquantizedLinearMethod._bart_original_apply = original_linear_apply
        UnquantizedLinearMethod._bart_batch_invariant_linear_patched = True

    if patch_lm_head and not getattr(
        UnquantizedEmbeddingMethod,
        "_bart_batch_invariant_linear_patched",
        False,
    ):
        original_embedding_apply = UnquantizedEmbeddingMethod.apply

        def embedding_apply(self, layer, x, bias=None):
            return linear_batch_invariant(x, layer.weight, bias)

        UnquantizedEmbeddingMethod.apply = embedding_apply
        UnquantizedEmbeddingMethod._bart_original_apply = original_embedding_apply
        UnquantizedEmbeddingMethod._bart_batch_invariant_linear_patched = True

    if (
        _BATCH_INVARIANT_LINEAR in scoped_model_modes
        or layer_scoped_mode
        or projection_scoped_mode
    ):
        from vllm_bart_plugin.bart import (
            BartCrossAttention,
            BartDecoder,
            BartDecoderLayer,
            BartDecoderSelfAttention,
            BartEncoder,
            BartEncoderAttention,
            BartEncoderLayer,
        )

        def layer_idx_from_prefix(prefix) -> int | None:
            if not isinstance(prefix, str):
                return None
            suffix = prefix.rsplit(".", 1)[-1]
            if not suffix.isdigit():
                return None
            return int(suffix)

        def with_linear_context(scope, fn, *args, **kwargs):
            token = _LINEAR_DIAG_CONTEXT.set(scope)
            try:
                return fn(*args, **kwargs)
            finally:
                _LINEAR_DIAG_CONTEXT.reset(token)

        if not getattr(BartEncoderLayer, "_bart_layer_index_patched", False):
            original_encoder_layer_init = BartEncoderLayer.__init__

            def encoder_layer_init(self, *args, **kwargs):
                original_encoder_layer_init(self, *args, **kwargs)
                self._bart_layer_idx = layer_idx_from_prefix(kwargs.get("prefix"))

            BartEncoderLayer.__init__ = encoder_layer_init
            BartEncoderLayer._bart_original_init = original_encoder_layer_init
            BartEncoderLayer._bart_layer_index_patched = True

        if not getattr(BartDecoderLayer, "_bart_layer_index_patched", False):
            original_decoder_layer_init = BartDecoderLayer.__init__

            def decoder_layer_init(self, *args, **kwargs):
                original_decoder_layer_init(self, *args, **kwargs)
                self._bart_layer_idx = layer_idx_from_prefix(kwargs.get("prefix"))
                self.encoder_attn._bart_layer_idx = self._bart_layer_idx

            BartDecoderLayer.__init__ = decoder_layer_init
            BartDecoderLayer._bart_original_init = original_decoder_layer_init
            BartDecoderLayer._bart_layer_index_patched = True

        if not getattr(BartEncoder, "_bart_linear_context_patched", False):
            original_encoder_forward = BartEncoder.forward

            def encoder_forward(self, *args, **kwargs):
                if _BATCH_INVARIANT_LINEAR != "encoder":
                    return original_encoder_forward(self, *args, **kwargs)
                return with_linear_context(
                    "encoder",
                    original_encoder_forward,
                    self,
                    *args,
                    **kwargs,
                )

            BartEncoder.forward = encoder_forward
            BartEncoder._bart_original_forward = original_encoder_forward
            BartEncoder._bart_linear_context_patched = True

        if not getattr(BartDecoder, "_bart_linear_context_patched", False):
            original_decoder_forward = BartDecoder.forward

            def decoder_forward(self, *args, **kwargs):
                if _BATCH_INVARIANT_LINEAR != "decoder":
                    return original_decoder_forward(self, *args, **kwargs)
                return with_linear_context(
                    "decoder",
                    original_decoder_forward,
                    self,
                    *args,
                    **kwargs,
                )

            BartDecoder.forward = decoder_forward
            BartDecoder._bart_original_forward = original_decoder_forward
            BartDecoder._bart_linear_context_patched = True

        if not getattr(BartEncoderAttention, "_bart_linear_context_patched", False):
            original_encoder_attention_forward = BartEncoderAttention.forward

            def encoder_attention_forward(self, *args, **kwargs):
                if _BATCH_INVARIANT_LINEAR != "encoder_attn":
                    return original_encoder_attention_forward(self, *args, **kwargs)
                return with_linear_context(
                    "encoder_attn",
                    original_encoder_attention_forward,
                    self,
                    *args,
                    **kwargs,
                )

            BartEncoderAttention.forward = encoder_attention_forward
            BartEncoderAttention._bart_original_forward = original_encoder_attention_forward
            BartEncoderAttention._bart_linear_context_patched = True

        if not getattr(BartDecoderSelfAttention, "_bart_linear_context_patched", False):
            original_decoder_self_attention_forward = BartDecoderSelfAttention.forward

            def decoder_self_attention_forward(self, *args, **kwargs):
                if _BATCH_INVARIANT_LINEAR != "decoder_self_attn":
                    return original_decoder_self_attention_forward(
                        self,
                        *args,
                        **kwargs,
                    )
                return with_linear_context(
                    "decoder_self_attn",
                    original_decoder_self_attention_forward,
                    self,
                    *args,
                    **kwargs,
                )

            BartDecoderSelfAttention.forward = decoder_self_attention_forward
            BartDecoderSelfAttention._bart_original_forward = (
                original_decoder_self_attention_forward
            )
            BartDecoderSelfAttention._bart_linear_context_patched = True

        if not getattr(BartCrossAttention, "_bart_linear_context_patched", False):
            original_cross_attention_forward = BartCrossAttention.forward

            def cross_attention_forward(
                self,
                decoder_hidden_states,
                encoder_hidden_states=None,
            ):
                if _BATCH_INVARIANT_LINEAR == "decoder_cross_attn":
                    return with_linear_context(
                        "decoder_cross_attn",
                        original_cross_attention_forward,
                        self,
                        decoder_hidden_states,
                        encoder_hidden_states,
                    )

                layer_idx = getattr(self, "_bart_layer_idx", None)
                q_layer = _linear_diag_layer_index("decoder_cross_attn_q_layer_")
                kv_layer = _linear_diag_layer_index("decoder_cross_attn_kv_layer_")
                out_layer = _linear_diag_layer_index("decoder_cross_attn_out_layer_")
                qkv_layer = _linear_diag_layer_index("decoder_cross_attn_qkv_layer_")
                qout_layer = _linear_diag_layer_index("decoder_cross_attn_qout_layer_")
                kvout_layer = _linear_diag_layer_index(
                    "decoder_cross_attn_kvout_layer_"
                )
                patch_q = (
                    q_layer == layer_idx
                    or qkv_layer == layer_idx
                    or qout_layer == layer_idx
                )
                patch_kv = (
                    kv_layer == layer_idx
                    or qkv_layer == layer_idx
                    or kvout_layer == layer_idx
                )
                patch_out = (
                    out_layer == layer_idx
                    or qout_layer == layer_idx
                    or kvout_layer == layer_idx
                )
                if not patch_q and not patch_kv and not patch_out:
                    return original_cross_attention_forward(
                        self,
                        decoder_hidden_states,
                        encoder_hidden_states,
                    )

                if patch_q:
                    q, _ = with_linear_context(
                        _BATCH_INVARIANT_LINEAR,
                        self.q_proj,
                        decoder_hidden_states,
                    )
                else:
                    q, _ = self.q_proj(decoder_hidden_states)

                if encoder_hidden_states is not None:
                    if patch_kv:
                        kv, _ = with_linear_context(
                            _BATCH_INVARIANT_LINEAR,
                            self.kv_proj,
                            encoder_hidden_states,
                        )
                    else:
                        kv, _ = self.kv_proj(encoder_hidden_states)
                    k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
                else:
                    k = v = None

                attn_output = self.attn(q, k, v)
                if patch_out:
                    output, _ = with_linear_context(
                        _BATCH_INVARIANT_LINEAR,
                        self.out_proj,
                        attn_output,
                    )
                else:
                    output, _ = self.out_proj(attn_output)
                return output

            BartCrossAttention.forward = cross_attention_forward
            BartCrossAttention._bart_original_forward = original_cross_attention_forward
            BartCrossAttention._bart_linear_context_patched = True

        if not getattr(BartEncoderLayer, "_bart_ffn_linear_context_patched", False):
            original_encoder_layer_forward = BartEncoderLayer.forward

            def encoder_layer_forward(self, hidden_states):
                layer_idx = getattr(self, "_bart_layer_idx", None)
                encoder_attn_layer = _linear_diag_layer_index(
                    "encoder_attn_layer_"
                )
                encoder_ffn_layer = _linear_diag_layer_index("encoder_ffn_layer_")
                patch_attn = (
                    _BATCH_INVARIANT_LINEAR == "encoder_attn"
                    or encoder_attn_layer == layer_idx
                )
                patch_ffn = (
                    _BATCH_INVARIANT_LINEAR == "encoder_ffn"
                    or encoder_ffn_layer == layer_idx
                )
                if not patch_attn and not patch_ffn:
                    return original_encoder_layer_forward(self, hidden_states)

                residual = hidden_states
                if patch_attn:
                    scope = (
                        _BATCH_INVARIANT_LINEAR
                        if encoder_attn_layer == layer_idx
                        else "encoder_attn"
                    )
                    hidden_states = with_linear_context(
                        scope,
                        self.self_attn,
                        hidden_states=hidden_states,
                    )
                else:
                    hidden_states = self.self_attn(hidden_states=hidden_states)
                hidden_states = residual + hidden_states
                hidden_states = self.self_attn_layer_norm(hidden_states)

                residual = hidden_states
                scope = (
                    _BATCH_INVARIANT_LINEAR
                    if encoder_ffn_layer == layer_idx
                    else "encoder_ffn"
                )
                token = (
                    _LINEAR_DIAG_CONTEXT.set(scope)
                    if patch_ffn
                    else None
                )
                try:
                    fc1_out, _ = self.fc1(hidden_states)
                    hidden_states = self.activation_fn(fc1_out)
                    hidden_states, _ = self.fc2(hidden_states)
                finally:
                    if token is not None:
                        _LINEAR_DIAG_CONTEXT.reset(token)

                hidden_states = residual + hidden_states
                hidden_states = self.final_layer_norm(hidden_states)
                return hidden_states

            BartEncoderLayer.forward = encoder_layer_forward
            BartEncoderLayer._bart_original_forward = original_encoder_layer_forward
            BartEncoderLayer._bart_ffn_linear_context_patched = True

        if not getattr(BartDecoderLayer, "_bart_ffn_linear_context_patched", False):
            original_decoder_layer_forward = BartDecoderLayer.forward

            def decoder_layer_forward(
                self,
                decoder_hidden_states,
                encoder_hidden_states=None,
            ):
                layer_idx = getattr(self, "_bart_layer_idx", None)
                decoder_self_attn_layer = _linear_diag_layer_index(
                    "decoder_self_attn_layer_"
                )
                decoder_cross_attn_layer = _linear_diag_layer_index(
                    "decoder_cross_attn_layer_"
                )
                decoder_ffn_layer = _linear_diag_layer_index("decoder_ffn_layer_")
                decoder_cross_ffn_layer = _linear_diag_layer_index(
                    "decoder_cross_ffn_layer_"
                )
                decoder_full_layer = _linear_diag_layer_index("decoder_layer_")
                patch_self_attn = (
                    _BATCH_INVARIANT_LINEAR == "decoder_self_attn"
                    or decoder_self_attn_layer == layer_idx
                    or decoder_full_layer == layer_idx
                )
                patch_cross_attn = (
                    _BATCH_INVARIANT_LINEAR == "decoder_cross_attn"
                    or decoder_cross_attn_layer == layer_idx
                    or decoder_cross_ffn_layer == layer_idx
                    or decoder_full_layer == layer_idx
                )
                patch_ffn = (
                    _BATCH_INVARIANT_LINEAR == "decoder_ffn"
                    or decoder_ffn_layer == layer_idx
                    or decoder_cross_ffn_layer == layer_idx
                    or decoder_full_layer == layer_idx
                )
                if not patch_self_attn and not patch_cross_attn and not patch_ffn:
                    return original_decoder_layer_forward(
                        self,
                        decoder_hidden_states,
                        encoder_hidden_states,
                    )

                residual = decoder_hidden_states
                if patch_self_attn:
                    scope = (
                        _BATCH_INVARIANT_LINEAR
                        if (
                            decoder_self_attn_layer == layer_idx
                            or decoder_full_layer == layer_idx
                        )
                        else "decoder_self_attn"
                    )
                    hidden_states = with_linear_context(
                        scope,
                        self.self_attn,
                        hidden_states=decoder_hidden_states,
                    )
                else:
                    hidden_states = self.self_attn(
                        hidden_states=decoder_hidden_states
                    )
                hidden_states = residual + hidden_states
                hidden_states = self.self_attn_layer_norm(hidden_states)

                residual = hidden_states
                if patch_cross_attn:
                    scope = (
                        _BATCH_INVARIANT_LINEAR
                        if (
                            decoder_cross_attn_layer == layer_idx
                            or decoder_cross_ffn_layer == layer_idx
                            or decoder_full_layer == layer_idx
                        )
                        else "decoder_cross_attn"
                    )
                    hidden_states = with_linear_context(
                        scope,
                        self.encoder_attn,
                        decoder_hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                else:
                    hidden_states = self.encoder_attn(
                        decoder_hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                    )
                hidden_states = residual + hidden_states
                hidden_states = self.encoder_attn_layer_norm(hidden_states)

                residual = hidden_states
                scope = (
                    _BATCH_INVARIANT_LINEAR
                    if (
                        decoder_ffn_layer == layer_idx
                        or decoder_cross_ffn_layer == layer_idx
                        or decoder_full_layer == layer_idx
                    )
                    else "decoder_ffn"
                )
                token = (
                    _LINEAR_DIAG_CONTEXT.set(scope)
                    if patch_ffn
                    else None
                )
                try:
                    fc1_out, _ = self.fc1(hidden_states)
                    hidden_states = self.activation_fn(fc1_out)
                    hidden_states, _ = self.fc2(hidden_states)
                finally:
                    if token is not None:
                        _LINEAR_DIAG_CONTEXT.reset(token)

                hidden_states = residual + hidden_states
                hidden_states = self.final_layer_norm(hidden_states)
                return hidden_states

            BartDecoderLayer.forward = decoder_layer_forward
            BartDecoderLayer._bart_original_forward = original_decoder_layer_forward
            BartDecoderLayer._bart_ffn_linear_context_patched = True


def install_runtime_patches() -> None:
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return

    _patch_encoder_overflow_check()
    _patch_batch_invariant_linear_diagnostic()
    _patch_openai_serving()
    _patch_bart_multimodal_processing()
    _patch_cross_attention_manager()
    _patch_hybrid_coordinator()
    _patch_cross_attention_backend()
    _patch_beam_scheduler()
    _patch_scheduler_encoder_handling()
    _patch_scheduler_free_tracing()
    _patch_kv_free_tracing()
    _patch_kv_allocate_encoder_tokens()
    _patch_gpu_model_runner_encoder_dedup()
    _patch_gpu_model_runner_encoder_tracing()
    _patch_gpu_model_runner_input_tracing()
    _patch_register_beam_search_logits_processor()
    _PATCHES_INSTALLED = True
    print(f"[BART_PLUGIN] Installed runtime patches (PID={os.getpid()})", flush=True)
