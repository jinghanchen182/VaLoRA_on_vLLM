# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import itertools
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Optional, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import (
    KVConnectorFactory)
from vllm.distributed.kv_transfer.kv_connector.v1 import (KVConnectorBase_V1,
                                                          KVConnectorRole)
from vllm.logger import init_logger
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (EncoderCacheManager,
                                                compute_encoder_budget)
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.core.sched.request_queue import (SchedulingPolicy,
                                              create_request_queue)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine import (EngineCoreEventType, EngineCoreOutput,
                            EngineCoreOutputs)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager

logger = init_logger(__name__)


class Scheduler(SchedulerInterface):

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.structured_output_manager = structured_output_manager

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: Optional[dict[int, set[str]]] = (
            defaultdict(set) if include_finished_set else None)

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = \
            self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.scheduler_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events)

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        if self.vllm_config.kv_transfer_config is not None:
            assert len(self.kv_cache_config.kv_cache_groups) == 1, (
                "Multiple KV cache groups are not currently supported "
                "with KV connectors")
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config, role=KVConnectorRole.SCHEDULER)

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        if self.scheduler_config.policy == "priority":
            self.policy = SchedulingPolicy.PRIORITY
        elif self.scheduler_config.policy == "fcfs":
            self.policy = SchedulingPolicy.FCFS
        else:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}")
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        # NOTE: For now we use the same budget for both compute and space.
        # This can be changed when we make encoder cache for embedding caching
        # across requests.
        encoder_compute_budget, encoder_cache_size = compute_encoder_budget(
            model_config=vllm_config.model_config,
            scheduler_config=vllm_config.scheduler_config,
            mm_registry=mm_registry,
        )

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed). Currently, we assume that the encoder also
        # has the Transformer architecture (e.g., ViT).
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = EncoderCacheManager(
            cache_size=encoder_cache_size)

        speculative_config = vllm_config.speculative_config

        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            caching_hash_algo=self.cache_config.prefix_caching_hash_algo,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
        )
        self.use_pp = self.parallel_config.pipeline_parallel_size > 1

        # 当前已经merge到模型权重中的lora_id, None表示没有merge任何lora
        self.current_merged_lora_id: Optional[int] = None

        # 额外调试字段：step 计数 & merge/unmerge 切换时间
        self.schedule_step_idx: int = 0
        self.merge_mode: str = "unmerge"
        self.last_merge_switch_ts: Optional[float] = None
        self.last_merge_switch_step: Optional[int] = None
        # TOS starvation tolerance threshold (seconds).
        self.tos_theta: float = 30.0
        # LOS scheduling hyper-parameters.
        self.los_alpha: float = 1.0
        self.los_beta: float = 0.1
        self.los_gamma: float = 0.5
        self.los_epsilon: float = 1.0
        self.los_phi_cache: float = 0.8

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        self.schedule_step_idx += 1

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        # NOTE: structured_output_request_ids maps
        # a request's (request that uses structured output)
        # request_id to the running request index.
        # This will helps us determine to slice the grammar bitmask
        # and only applies valid mask for requests that
        # uses structured decoding.
        structured_output_request_ids: dict[str, int] = {}

        req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        merge_lora_id = None
        lora_id_to_unmerge = None
        lora_id_to_merge = None
        tos_starve_req_ids: set[str] = set()
        tos_dom_lora_id: Optional[int] = None
        tos_t_starve = 0
        tos_t_dom = 0
        los_score_by_req_id: dict[str, float] = {}

        # Select LoRA scheduling backend via environment variable.
        # Options: "baseline" (pure Punica), "simple", "TOS", "LOS"
        lora_backend = os.environ.get("VLLM_LORA_BACKEND", "baseline")
        if lora_backend == "simple":
            old_merge_mode = getattr(self, "merge_mode", "unmerge")

            # 统计running队列的lora id
            running_lora_id_counts = {}
            for req in self.running:
                lora_req = getattr(req, "lora_request", None)
                if lora_req is not None:
                    lora_id = getattr(lora_req, "lora_int_id", None)
                    if lora_id is not None:
                        running_lora_id_counts[lora_id] = running_lora_id_counts.get(lora_id, 0) + 1
            
            # 统计waiting队列的lora id
            waiting_lora_id_counts = {}
            for req in self.waiting:
                lora_req = getattr(req, "lora_request", None)
                if lora_req is not None:
                    lora_id = getattr(lora_req, "lora_int_id", None)
                    if lora_id is not None:
                        waiting_lora_id_counts[lora_id] = waiting_lora_id_counts.get(lora_id, 0) + 1
            
            # 合并所有请求的lora id统计
            all_lora_id_counts = {}
            for lora_id, count in running_lora_id_counts.items():
                all_lora_id_counts[lora_id] = all_lora_id_counts.get(lora_id, 0) + count
            for lora_id, count in waiting_lora_id_counts.items():
                all_lora_id_counts[lora_id] = all_lora_id_counts.get(lora_id, 0) + count
            
            # logger.info(f"[TEST] Valora Scheduling Step Start")
            # 调试：观测 waiting 队列中 L1/L2 分布，便于判断是否存在 有 L2 但当前在处理 L1 chunk
            if all_lora_id_counts:
                logger.info(
                    "[TEST] step=%d queue_lora_dist: "
                    "running=%s, waiting=%s, all=%s, "
                    "current_merged_lora_id=%s, merge_mode=%s",
                    self.schedule_step_idx,
                    running_lora_id_counts,
                    waiting_lora_id_counts,
                    all_lora_id_counts,
                    self.current_merged_lora_id,
                    getattr(self, "merge_mode", "unmerge"),
                )

            # 判断是否应该进入merge模式
            # 1. 如果所有请求的lora id相同且只有一个lora_id
            # 2. 或者某个lora_id的请求数超过一半
            if not all_lora_id_counts:
                # 没有任何 LoRA 请求，保持现状或回退到 unmerge
                self.merge_mode = "merge" if self.current_merged_lora_id is not None else "unmerge"
                merge_lora_id = self.current_merged_lora_id
            elif len(running_lora_id_counts) > 1:
                self.merge_mode = "unmerge"
            elif len(all_lora_id_counts) == 1:
                # 只有一个lora_id，直接使用它
                merge_lora_id = list(all_lora_id_counts.keys())[0]
                self.merge_mode = "merge"
            else:
                # 多个lora_id，检查是否有某个超过一半
                total_reqs = sum(all_lora_id_counts.values())
                dominant_lora = None
                for lora_id, count in all_lora_id_counts.items():
                    if count > total_reqs / 2:
                        dominant_lora = lora_id
                        break
                if dominant_lora is not None:
                    merge_lora_id = dominant_lora
                    self.merge_mode = "merge"
                else:
                    self.merge_mode = "unmerge"

            # 处理merge模式下的lora权重管理
            if self.merge_mode == "merge" and merge_lora_id is not None:
                # 如果当前已经merge了其他lora，需要先unmerge
                if self.current_merged_lora_id is not None and \
                self.current_merged_lora_id != merge_lora_id:
                    lora_id_to_unmerge = self.current_merged_lora_id
                    # logger.info(f"需要unmerge lora_id: {lora_id_to_unmerge}")
                
                # 如果新的merge_lora_id还没有merge到模型，需要merge
                if self.current_merged_lora_id != merge_lora_id:
                    lora_id_to_merge = merge_lora_id
                    # logger.info(f"需要merge lora_id: {lora_id_to_merge}")
                    # 更新当前merged的lora_id
                    self.current_merged_lora_id = merge_lora_id
            elif self.merge_mode == "unmerge":
                # 如果切换到unmerge模式，但当前有merge的lora，需要unmerge
                if self.current_merged_lora_id is not None:
                    lora_id_to_unmerge = self.current_merged_lora_id
                    self.current_merged_lora_id = None

            # 如果本次 step 发生了模式切换，打 log 并记录切换时间
            if self.merge_mode != old_merge_mode:
                now_ts = scheduled_timestamp
                delta = None
                if self.last_merge_switch_ts is not None:
                    delta = now_ts - self.last_merge_switch_ts
                self.last_merge_switch_ts = now_ts
                self.last_merge_switch_step = self.schedule_step_idx

                logger.info(
                    "[TEST] step=%d merge_mode_switch: %s -> %s, "
                    "merge_lora_id=%s, last_switch_delta=%.6f",
                    self.schedule_step_idx,
                    old_merge_mode,
                    self.merge_mode,
                    merge_lora_id,
                    float(delta) if delta is not None else -1.0,
                )

                if lora_id_to_unmerge is not None or lora_id_to_merge is not None:
                    logger.info(
                        "[TEST] step=%d merge_ops: to_unmerge=%s, to_merge=%s",
                        self.schedule_step_idx,
                        lora_id_to_unmerge,
                        lora_id_to_merge,
                    )
        elif lora_backend  == "TOS":
            old_merge_mode = getattr(self, "merge_mode", "unmerge")
            max_t = max(1, self.max_num_scheduled_tokens)
            now_wall_ts = time.time()

            all_reqs = [*self.running, *list(self.waiting)]
            lora_token_demands: dict[int, int] = defaultdict(int)

            for req in all_reqs:
                # Under chunked prefill, use remaining token demand as workload
                # granularity instead of request counts.
                if req.status == RequestStatus.RUNNING:
                    token_demand = max(
                        0,
                        req.num_tokens_with_spec + req.num_output_placeholders -
                        req.num_computed_tokens,
                    )
                else:
                    token_demand = max(0, req.num_tokens -
                                       req.num_computed_tokens)

                lora_req = getattr(req, "lora_request", None)
                req_lora_id = getattr(lora_req, "lora_int_id",
                                      None) if lora_req else None
                if req_lora_id is not None and token_demand > 0:
                    lora_token_demands[req_lora_id] += token_demand

                waiting_time = max(0.0, now_wall_ts - req.arrival_time)
                # Approximate current-mode execution time using token demand
                # normalized by MaxT, and add recent mode-switch latency.
                exec_time_in_mode = token_demand / max_t
                switch_latency = 0.0
                if self.last_merge_switch_ts is not None:
                    switch_latency = max(
                        0.0, scheduled_timestamp - self.last_merge_switch_ts)
                credit = waiting_time + exec_time_in_mode + switch_latency
                if credit > self.tos_theta:
                    tos_starve_req_ids.add(req.request_id)
                    tos_t_starve += token_demand

            if lora_token_demands:
                tos_dom_lora_id, tos_t_dom = max(
                    lora_token_demands.items(),
                    key=lambda item: item[1],
                )
                merge_lora_id = tos_dom_lora_id

            if (tos_t_starve / max_t <= 0.5) and (tos_t_dom / max_t > 0.5):
                if tos_t_starve == 0 and tos_dom_lora_id is not None:
                    self.merge_mode = "merge"
                elif tos_dom_lora_id is not None:
                    self.merge_mode = "mix"
                else:
                    self.merge_mode = "unmerge"
            else:
                self.merge_mode = "unmerge"

            if self.merge_mode in ("merge", "mix") and merge_lora_id is not None:
                if (self.current_merged_lora_id is not None
                        and self.current_merged_lora_id != merge_lora_id):
                    lora_id_to_unmerge = self.current_merged_lora_id

                if self.current_merged_lora_id != merge_lora_id:
                    lora_id_to_merge = merge_lora_id
                    self.current_merged_lora_id = merge_lora_id
            elif self.merge_mode == "unmerge":
                if self.current_merged_lora_id is not None:
                    lora_id_to_unmerge = self.current_merged_lora_id
                    self.current_merged_lora_id = None

            logger.info(
                "[TEST][TOS] step=%d mode_decision: mode=%s, "
                "t_starve=%d(%.3f), t_dom=%d(%.3f), max_t=%d, "
                "starve_cnt=%d, dom_lora_id=%s",
                self.schedule_step_idx,
                self.merge_mode,
                tos_t_starve,
                tos_t_starve / max_t,
                tos_t_dom,
                tos_t_dom / max_t,
                max_t,
                len(tos_starve_req_ids),
                tos_dom_lora_id,
            )

            if self.merge_mode != old_merge_mode:
                now_ts = scheduled_timestamp
                delta = None
                if self.last_merge_switch_ts is not None:
                    delta = now_ts - self.last_merge_switch_ts
                self.last_merge_switch_ts = now_ts
                self.last_merge_switch_step = self.schedule_step_idx

                logger.info(
                    "[TEST][TOS] step=%d merge_mode_switch: %s -> %s, "
                    "dom_lora_id=%s, last_switch_delta=%.6f",
                    self.schedule_step_idx,
                    old_merge_mode,
                    self.merge_mode,
                    tos_dom_lora_id,
                    float(delta) if delta is not None else -1.0,
                )

                if lora_id_to_unmerge is not None or lora_id_to_merge is not None:
                    logger.info(
                        "[TEST][TOS] step=%d merge_ops: to_unmerge=%s, "
                        "to_merge=%s",
                        self.schedule_step_idx,
                        lora_id_to_unmerge,
                        lora_id_to_merge,
                    )
        elif lora_backend == "LOS":
            old_merge_mode = getattr(self, "merge_mode", "unmerge")
            now_wall_ts = time.time()

            if self.waiting:
                los_ordered_waiting: list[Request] = []
                while self.waiting:
                    request = self.waiting.pop_request()
                    l_prompt = max(1, request.num_prompt_tokens)
                    l_hit = request.num_cached_tokens
                    if l_hit < 0:
                        _, num_new_local_computed_tokens = (
                            self.kv_cache_manager.get_computed_blocks(request))
                        l_hit = num_new_local_computed_tokens
                    l_hit = min(max(0, l_hit), l_prompt)
                    t_wait = max(0.0, now_wall_ts - request.arrival_time)
                    lora_req = getattr(request, "lora_request", None)
                    req_lora_id = getattr(lora_req, "lora_int_id",
                                          None) if lora_req else None
                    merged_bonus = 1.0 if (
                        req_lora_id is not None
                        and req_lora_id == self.current_merged_lora_id) else 0.0
                    uncached_len = max(0.0,
                                       float(l_prompt - l_hit) +
                                       self.los_epsilon)
                    score = (self.los_alpha * (1.0 / uncached_len) +
                             self.los_beta * t_wait +
                             self.los_gamma * merged_bonus)
                    los_score_by_req_id[request.request_id] = score
                    los_ordered_waiting.append(request)

                los_ordered_waiting.sort(
                    key=lambda req: los_score_by_req_id.get(req.request_id, 0.0),
                    reverse=True,
                )
                for request in reversed(los_ordered_waiting):
                    self.waiting.prepend_request(request)

            # LOS decides mode after the batch is assembled.
            self.merge_mode = old_merge_mode
        else:
            logger.info(f"[TEST] Punica Scheduling Step Start")
        
        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            num_new_tokens = (request.num_tokens_with_spec +
                              request.num_output_placeholders -
                              request.num_computed_tokens)
            if (0 < self.scheduler_config.long_prefill_token_threshold <
                    num_new_tokens):
                num_new_tokens = (
                    self.scheduler_config.long_prefill_token_threshold)
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens,
                self.max_model_len - 1 - request.num_computed_tokens)

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            new_encoder_budget = encoder_budget
            if request.has_encoder_inputs:
                (encoder_inputs_to_schedule, num_new_tokens,
                 new_encoder_budget) = self._try_schedule_encoder_inputs(
                     request, request.num_computed_tokens, num_new_tokens,
                     encoder_budget)

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            while True:
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                    num_lookahead_tokens=self.num_lookahead_tokens)
                if new_blocks is None:
                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                    else:
                        preempted_req = self.running.pop()

                    self.kv_cache_manager.free(preempted_req)
                    preempted_req.status = RequestStatus.PREEMPTED
                    preempted_req.num_computed_tokens = 0
                    if self.log_stats:
                        preempted_req.record_event(
                            EngineCoreEventType.PREEMPTED, scheduled_timestamp)

                    self.waiting.prepend_request(preempted_req)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt.
                        can_schedule = False
                        break
                else:
                    # The request can be scheduled.
                    can_schedule = True
                    break
            if not can_schedule:
                break
            assert new_blocks is not None

            # Schedule the request.
            scheduled_running_reqs.append(request)
            if request.use_structured_output:
                # PERF: in case of chunked prefill,
                # request might not include any new tokens.
                # Therefore, we might introduce some additional
                # cycle to fill in the bitmask, which could be a big no-op.
                structured_output_request_ids[request.request_id] = req_index
            req_to_new_block_ids[request.request_id] = (
                new_blocks.get_block_ids())
            num_scheduled_tokens[request.request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (num_new_tokens +
                                             request.num_computed_tokens -
                                             request.num_tokens)
                if num_scheduled_spec_tokens > 0:
                    # Trim spec_token_ids list to num_scheduled_spec_tokens.
                    del request.spec_token_ids[num_scheduled_spec_tokens:]
                    scheduled_spec_decode_tokens[request.request_id] = (
                        request.spec_token_ids)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request.request_id] = (
                    encoder_inputs_to_schedule)
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                encoder_budget = new_encoder_budget

            # logger.info(f"[TEST] Scheduling RUNNING Request {request.request_id}: num_new_tokens={num_new_tokens}, token_budget={token_budget}, encoder_budget={encoder_budget}")
            # logger.info(f"[TEST] Process : {request.num_computed_tokens + num_new_tokens} / {request.num_tokens_with_spec}")

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0)
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Use a temporary RequestQueue to collect requests that need to be
        # skipped and put back at the head of the waiting queue later
        skipped_waiting_requests = create_request_queue(self.policy)

        
        # Next, schedule the WAITING requests.
        if not preempted_reqs:
            if lora_backend == "TOS" and tos_starve_req_ids and self.waiting:
                starving_waiting: list[Request] = []
                non_starving_waiting: list[Request] = []
                while self.waiting:
                    waiting_req = self.waiting.pop_request()
                    if waiting_req.request_id in tos_starve_req_ids:
                        starving_waiting.append(waiting_req)
                    else:
                        non_starving_waiting.append(waiting_req)
                for waiting_req in reversed(non_starving_waiting):
                    self.waiting.prepend_request(waiting_req)
                for waiting_req in reversed(starving_waiting):
                    self.waiting.prepend_request(waiting_req)

            while self.waiting and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request = self.waiting.peek_request()

                # merge模式下, 跳过不匹配merge_lora_id的请求
                if lora_backend == "simple" and self.merge_mode == "merge":
                    lora_req = getattr(request, "lora_request", None)
                    req_lora_id = getattr(lora_req, "lora_int_id", None) if lora_req else None

                    if req_lora_id != merge_lora_id:
                        # 跳过该请求
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                if lora_backend == "TOS":
                    lora_req = getattr(request, "lora_request", None)
                    req_lora_id = getattr(lora_req, "lora_int_id",
                                          None) if lora_req else None
                    if (self.merge_mode == "merge"
                            and req_lora_id != tos_dom_lora_id):
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue
                    if self.merge_mode == "mix":
                        is_starving = request.request_id in tos_starve_req_ids
                        is_dominant = req_lora_id == tos_dom_lora_id
                        if not (is_starving or is_dominant):
                            self.waiting.pop_request()
                            skipped_waiting_requests.prepend_request(request)
                            continue

                # KVTransfer: skip request if still waiting for remote kvs.
                if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                    is_ready = self._update_waiting_for_remote_kv(request)
                    if is_ready:
                        request.status = RequestStatus.WAITING
                    else:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request.request_id)
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Skip request if the structured output request is still waiting
                # for FSM compilation.
                if request.status == RequestStatus.WAITING_FOR_FSM:
                    structured_output_req = request.structured_output_request
                    if structured_output_req and structured_output_req.grammar:
                        request.status = RequestStatus.WAITING
                    else:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (self.lora_config and request.lora_request and
                    (len(scheduled_loras) == self.lora_config.max_loras and
                     request.lora_request.lora_int_id not in scheduled_loras)):
                    # Scheduling would exceed max_loras, skip.
                    self.waiting.pop_request()
                    skipped_waiting_requests.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = \
                        self.kv_cache_manager.get_computed_blocks(
                            request)

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        num_external_computed_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens))

                    # Total computed tokens (local + external).
                    num_computed_tokens = (num_new_local_computed_tokens +
                                           num_external_computed_tokens)
                # KVTransfer: WAITING reqs have num_computed_tokens > 0
                # after async KV recvs are completed.
                else:
                    new_computed_blocks = (
                        self.kv_cache_manager.create_empty_block_list())
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                new_encoder_budget = encoder_budget

                # KVTransfer: loading remote KV, do not allocate for new work.
                if load_kv_async:
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                # Number of tokens to be scheduled.
                else:
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    if (0 < self.scheduler_config.long_prefill_token_threshold
                            < num_new_tokens):
                        num_new_tokens = (
                            self.scheduler_config.long_prefill_token_threshold)

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if not self.scheduler_config.chunked_prefill_enabled and \
                        num_new_tokens > token_budget:
                        self.waiting.pop_request()
                        skipped_waiting_requests.prepend_request(request)
                        continue

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (encoder_inputs_to_schedule, num_new_tokens,
                         new_encoder_budget
                         ) = self._try_schedule_encoder_inputs(
                             request, num_computed_tokens, num_new_tokens,
                             encoder_budget)
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (0 if request.num_computed_tokens
                                              == 0 else
                                              self.num_lookahead_tokens)

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens + num_external_computed_tokens,
                    num_new_local_computed_tokens,
                    new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    delay_cache_blocks=load_kv_async,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        new_computed_blocks + new_blocks,
                        num_external_computed_tokens,
                    )

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = self.waiting.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    skipped_waiting_requests.prepend_request(request)
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    continue

                if request.use_structured_output:
                    structured_output_request_ids[request.request_id] = (
                        req_index)
                req_index += 1
                self.running.append(request)
                if self.log_stats:
                    request.record_event(EngineCoreEventType.SCHEDULED,
                                         scheduled_timestamp)
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(
                        f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_block_ids[request.request_id] = (
                    self.kv_cache_manager.get_block_ids(request.request_id))
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request.request_id] = (
                        encoder_inputs_to_schedule)
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                    encoder_budget = new_encoder_budget

        # Put back any skipped requests at the head of the waiting queue
        if skipped_waiting_requests:
            self.waiting.prepend_requests(skipped_waiting_requests)

        if lora_backend == "LOS":
            old_merge_mode = self.merge_mode
            scheduled_batch_reqs = [
                *scheduled_running_reqs,
                *scheduled_new_reqs,
                *scheduled_resumed_reqs,
            ]
            total_prompt_tokens = 0
            total_hit_tokens = 0
            los_lora_counts: dict[int, int] = defaultdict(int)

            for req in scheduled_batch_reqs:
                l_prompt = max(1, req.num_prompt_tokens)
                l_hit = min(max(0, req.num_cached_tokens), l_prompt)
                total_prompt_tokens += l_prompt
                total_hit_tokens += l_hit
                lora_req = getattr(req, "lora_request", None)
                req_lora_id = getattr(lora_req, "lora_int_id",
                                      None) if lora_req else None
                if req_lora_id is not None:
                    los_lora_counts[req_lora_id] += 1

            ratio_hit = (
                total_hit_tokens / total_prompt_tokens
                if total_prompt_tokens > 0 else 0.0)
            unique_lora_num = len(los_lora_counts)
            los_dominant_lora_id: Optional[int] = None
            if los_lora_counts:
                los_dominant_lora_id = max(
                    los_lora_counts.items(),
                    key=lambda item: item[1],
                )[0]

            if unique_lora_num > 1 and ratio_hit > self.los_phi_cache:
                self.merge_mode = "unmerge"
                merge_lora_id = None
            elif unique_lora_num == 1 and los_dominant_lora_id is not None:
                self.merge_mode = "merge"
                merge_lora_id = los_dominant_lora_id
            elif unique_lora_num > 1 and los_dominant_lora_id is not None:
                self.merge_mode = "mix"
                merge_lora_id = los_dominant_lora_id

            if self.merge_mode in ("merge", "mix") and merge_lora_id is not None:
                if (self.current_merged_lora_id is not None
                        and self.current_merged_lora_id != merge_lora_id):
                    lora_id_to_unmerge = self.current_merged_lora_id
                if self.current_merged_lora_id != merge_lora_id:
                    lora_id_to_merge = merge_lora_id
                    self.current_merged_lora_id = merge_lora_id
            elif self.merge_mode == "unmerge":
                if self.current_merged_lora_id is not None:
                    lora_id_to_unmerge = self.current_merged_lora_id
                    self.current_merged_lora_id = None
                    merge_lora_id = None

            logger.info(
                "[TEST][LOS] step=%d mode_decision: mode=%s, "
                "ratio_hit=%.3f, unique_loras=%d, dominant_lora_id=%s, "
                "phi_cache=%.3f",
                self.schedule_step_idx,
                self.merge_mode,
                ratio_hit,
                unique_lora_num,
                los_dominant_lora_id,
                self.los_phi_cache,
            )

            if self.merge_mode != old_merge_mode:
                delta = None
                if self.last_merge_switch_ts is not None:
                    delta = scheduled_timestamp - self.last_merge_switch_ts
                self.last_merge_switch_ts = scheduled_timestamp
                self.last_merge_switch_step = self.schedule_step_idx
                logger.info(
                    "[TEST][LOS] step=%d merge_mode_switch: %s -> %s, "
                    "dominant_lora_id=%s, last_switch_delta=%.6f",
                    self.schedule_step_idx,
                    old_merge_mode,
                    self.merge_mode,
                    los_dominant_lora_id,
                    float(delta) if delta is not None else -1.0,
                )

                if lora_id_to_unmerge is not None or lora_id_to_merge is not None:
                    logger.info(
                        "[TEST][LOS] step=%d merge_ops: to_unmerge=%s, "
                        "to_merge=%s",
                        self.schedule_step_idx,
                        lora_id_to_unmerge,
                        lora_id_to_merge,
                    )

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens
        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert (len(scheduled_new_reqs) + len(scheduled_resumed_reqs) +
                len(scheduled_running_reqs) <= len(self.running))

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(
            self.kv_cache_config.kv_cache_groups)
        if self.running:
            any_request = self.running[0]
            num_common_prefix_blocks = (
                self.kv_cache_manager.get_num_common_prefix_blocks(
                    any_request, len(self.running)))

        grammar_bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduled_spec_decode_tokens,
        )

        if num_scheduled_tokens:
            step_reqs_info = []
            for req_id, n_tok in num_scheduled_tokens.items():
                req = self.requests.get(req_id)
                if req is None:
                    continue
                lora_req = getattr(req, "lora_request", None)
                lora_id = getattr(lora_req, "lora_int_id", None) if lora_req else None
                step_reqs_info.append(
                    f"(req_id={req_id}, lora_id={lora_id}, tokens={n_tok})"
                )

            logger.info(
                "[TEST] step=%d scheduled_reqs: %s, "
                "merge_mode=%s, merged_lora_id=%s, total_tokens=%d",
                self.schedule_step_idx,
                "; ".join(step_reqs_info),
                self.merge_mode,
                self.current_merged_lora_id,
                total_num_scheduled_tokens,
            )

        # Construct the scheduler output.
        new_reqs_data = [
            NewRequestData.from_request(req,
                                        req_to_new_block_ids[req.request_id])
            for req in scheduled_new_reqs
        ]
        cached_reqs_data = self._make_cached_request_data(
            scheduled_running_reqs,
            scheduled_resumed_reqs,
            num_scheduled_tokens,
            scheduled_spec_decode_tokens,
            req_to_new_block_ids,
        )
        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_input_ids=self.encoder_cache_manager.get_freed_ids(),
            structured_output_request_ids=structured_output_request_ids,
            grammar_bitmask=grammar_bitmask,
            # LoRA merge相关信息
            lora_id_to_unmerge=lora_id_to_unmerge,
            lora_id_to_merge=lora_id_to_merge,
            lora_infer_mode=self.merge_mode,
            mix_primary_lora_id=(merge_lora_id
                                 if self.merge_mode == "mix" else None),
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self.connector.build_connector_meta(scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        events = self.kv_cache_manager.take_events()
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _update_after_schedule(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_block_ids: dict[str, tuple[list[int], ...]],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...]] = []
        num_computed_tokens: list[int] = []

        use_connector = self.connector is not None
        for req in itertools.chain(running_reqs, resumed_reqs):
            req_id = req.request_id
            req_ids.append(req_id)
            num_tokens = (num_scheduled_tokens[req_id] -
                          len(spec_decode_tokens.get(req_id, ())))
            if self.use_pp:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                token_ids = req.all_token_ids[req.num_computed_tokens:req.
                                              num_computed_tokens + num_tokens]
                new_token_ids.append(token_ids)
            elif use_connector:
                # When using a KVConnector, we add a placeholder to avoid index
                # out of bounds errors. TODO: Remove this once the KVConnector
                # is updated to handle token IDs properly.
                new_token_ids.append([])
            new_block_ids.append(req_to_new_block_ids[req_id])
            num_computed_tokens.append(req.num_computed_tokens)
        # Because resumed_reqs is usually empty, it is more efficient to do
        # in-place appending so that we don't need to allocate a new list.
        resumed_from_preemption = [False] * len(running_reqs)
        resumed_from_preemption += [True] * len(resumed_reqs)

        return CachedRequestData(
            req_ids=req_ids,
            resumed_from_preemption=resumed_from_preemption,
            new_token_ids=new_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_budget: int,
    ) -> tuple[list[int], int, int]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_budget
        encoder_inputs_to_schedule: list[int] = []
        mm_positions = request.mm_positions
        assert mm_positions is not None
        assert len(mm_positions) > 0
        for i, pos_info in enumerate(mm_positions):
            start_pos = pos_info.offset
            num_encoder_tokens = pos_info.length

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if start_pos >= num_computed_tokens + num_new_tokens:
                # The encoder input is not needed in this step.
                break
            if start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if self.encoder_cache_manager.has_cache(request, i):
                # The encoder input is already computed and cached.
                continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (self.scheduler_config.disable_chunked_mm_input
                    and num_computed_tokens < start_pos
                    and (num_computed_tokens + num_new_tokens)
                    < (start_pos + num_encoder_tokens)):
                num_new_tokens = start_pos - num_computed_tokens
                break

            if (not self.encoder_cache_manager.can_allocate(request, i)
                    or num_encoder_tokens > encoder_budget):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - num_computed_tokens
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            encoder_budget -= num_encoder_tokens
            encoder_inputs_to_schedule.append(i)
        return encoder_inputs_to_schedule, num_new_tokens, encoder_budget

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        spec_token_ids = model_runner_output.spec_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: Optional[SpecDecodingStats] = None

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            request = self.requests.get(req_id)
            if request is None:
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism).
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = sampled_token_ids[
                req_index] if sampled_token_ids else []

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id))
            if scheduled_spec_token_ids:
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens, where is given by:
                # len(scheduled_spec_token_ids) + 1 - len(generated_token_ids).
                num_tokens_rejected = (len(scheduled_spec_token_ids) + 1 -
                                       len(generated_token_ids))
                request.num_computed_tokens -= num_tokens_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=len(scheduled_spec_token_ids),
                    num_accepted_tokens=len(generated_token_ids) - 1)

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids)

            # Stop checking for pooler models.
            pooler_output = None
            if pooler_outputs:
                pooler_output = pooler_outputs[req_index]
                stopped = check_stop(request, self.max_model_len,
                                     pooler_output)

            if stopped:
                kv_transfer_params = self._free_request(request)
                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if request.sampling_params is not None \
                and request.sampling_params.logprobs is not None and logprobs:
                # NOTE: once we support N tokens per step (spec decode),
                # the outer lists can be of length > 1.
                new_logprobs = logprobs.slice(req_index, req_index + 1)

            if new_token_ids and self.structured_output_manager.should_advance(
                    request):
                # NOTE: structured_output_request
                # should not be None if use_structured_output, we have
                # check above, so safe to ignore type warning
                request.structured_output_request.grammar.accept_tokens(  # type: ignore[union-attr]
                    req_id, new_token_ids)

            # spec_token_ids comes from the model runner output
            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Add newly generated spec token ids to the request.
            if spec_token_ids is not None:
                if self.structured_output_manager.should_advance(request):
                    metadata = request.structured_output_request
                    # Needs to happen after new_token_ids are accepted.
                    request.spec_token_ids = metadata.grammar.validate_tokens(  # type: ignore[union-attr]
                        spec_token_ids[req_index])
                else:
                    request.spec_token_ids = spec_token_ids[req_index]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if new_token_ids or pooler_output is not None \
                or kv_transfer_params:

                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=request.get_finished_reason(),
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        num_cached_tokens=request.num_cached_tokens,
                    ))

            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = [
                req for req in self.running if req not in stopped_running_reqs
            ]
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        # KV Connector: update state for finished KV Transfers.
        if model_runner_output.kv_connector_output:
            self._update_from_kv_xfer_finished(
                model_runner_output.kv_connector_output)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set)
            finished_req_ids.clear()

        if engine_core_outputs:
            # Return stats to only one of the front-ends.
            next(iter(engine_core_outputs.values())).scheduler_stats = (
                self.make_stats(spec_decoding_stats))

            # 调试：如果最近发生了 merge_mode 切换，记录从那次切换到本 batch 完成的耗时
            if self.last_merge_switch_ts is not None and self.last_merge_switch_step is not None:
                now_ts = time.monotonic()
                logger.info(
                    "[TEST] batch_done_after_switch: "
                    "switch_step=%d, now_step_approx=%d, "
                    "elapsed_since_switch=%.6f s, num_outputs_clients=%d",
                    self.last_merge_switch_step,
                    self.last_merge_switch_step + 1,  # 近似对应的下一个 step
                    now_ts - self.last_merge_switch_ts,
                    len(engine_core_outputs),
                )

        return engine_core_outputs

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = (
            self.encoder_cache_manager.get_cached_input_ids(request))
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_positions = request.mm_positions[input_id]
            start_pos = mm_positions.offset
            num_tokens = mm_positions.length
            if start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(
                    request, input_id)

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting)

    def add_request(self, request: Request) -> None:
        self.waiting.add_request(request)
        self.requests[request.request_id] = request
        if self.log_stats:
            request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: RequestStatus,
    ) -> None:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids, )
        else:
            request_ids = set(request_ids)

        running_requests_to_remove = []
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None:
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.append(request)
            else:
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        for request in running_requests_to_remove:
            self.running.remove(request)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            request.status = finished_status
            self._free_request(request)

    def _free_request(self, request: Request) -> Optional[dict[str, Any]]:
        assert request.is_finished()

        delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        self.kv_cache_manager.free_block_hashes(request)
        del self.requests[request.request_id]

    def get_num_unfinished_requests(self) -> int:
        return len(self.waiting) + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(self) -> bool:
        return self.kv_cache_manager.reset_prefix_cache()

    def make_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats] = None,
    ) -> Optional[SchedulerStats]:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            prefix_cache_stats=prefix_cache_stats,
            spec_decoding_stats=spec_decoding_stats,
            num_corrupted_reqs=sum(req.is_output_corrupted
                                   for req in self.running),
        )

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: Optional[SpecDecodingStats],
        num_draft_tokens: int,
        num_accepted_tokens: int,
    ) -> Optional[SpecDecodingStats]:
        if not self.log_stats:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens,
            num_accepted_tokens=num_accepted_tokens)
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    def _connector_finished(
            self, request: Request) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        return self.connector.request_finished(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> bool:
        """
        KV Connector: check if the request_id is finished_recving.

        The finished_recving_kv_req_ids list is populated
        on the previous steps()'s update_from_output based
        on the worker side connector.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None
        if request.request_id not in self.finished_recving_kv_req_ids:
            return False

        # Now that the blocks are ready, actually cache them.
        (block_ids, ) = self.kv_cache_manager.get_block_ids(request.request_id)
        num_computed_tokens = len(block_ids) * self.block_size
        # Handle the case where num request tokens less then one block.
        num_computed_tokens = min(num_computed_tokens, request.num_tokens)
        if num_computed_tokens == request.num_tokens:
            num_computed_tokens -= 1
        # This will cache the blocks iff caching is enabled.
        self.kv_cache_manager.cache_blocks(request, num_computed_tokens)

        # Update the request state for scheduling.
        request.num_computed_tokens = num_computed_tokens

        # Return that we are ready.
        self.finished_recving_kv_req_ids.remove(request.request_id)
        return True

    def _update_from_kv_xfer_finished(self,
                                      kv_connector_output: KVConnectorOutput):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            scheduler the request during the next step.
        """

        if self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        # KV Connector:: update recv and send status from last step.
        for req_id in (kv_connector_output.finished_recving or ()):
            logger.debug("Finished recving KV transfer for request %s", req_id)
            self.finished_recving_kv_req_ids.add(req_id)
        for req_id in (kv_connector_output.finished_sending or ()):
            logger.debug("Finished sending KV transfer for request %s", req_id)
            self._free_blocks(self.requests[req_id])
