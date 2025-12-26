# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Based on:
Chen, L., Ye, Z., Wu, Y., Zhuo, D., Ceze, L., & Krishnamurthy, A. (2023). 
Punica: Multi-Tenant LoRA Serving. 
https://arxiv.org/abs/2310.18547
"""

from typing import Optional, Union, final

import torch

import vllm.envs as envs
from vllm.lora.layers import LoRAMapping
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    from vllm.lora.ops.triton_ops import (LoRAKernelMeta, lora_expand,
                                          lora_shrink)

from .punica_base import PunicaWrapperBase

from vllm.logger import init_logger
logger = init_logger(__name__)

@final
class PunicaWrapperGPU(PunicaWrapperBase):
    """
    PunicaWrapperGPU is designed to manage and provide metadata for the punica 
    kernel. The main function is to maintain the state information for 
    Multi-LoRA, and to provide the interface for the punica triton kernel.
    """

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        self.max_loras = kwargs['max_loras']

        self.token_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                      max_num_batched_tokens,
                                                      device=device)

        # When cudagraph capture size is greater than max_num_seqs (max_batches,
        # here), V0 captures the graph as if max_num_seqs is set to
        # the capture size.
        # V1 doesn't have this problem and always respects max_num_seqs.
        max_num_prompts = (max_batches
                           if envs.VLLM_USE_V1 else max_num_batched_tokens)
        self.prompt_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                       max_num_prompts,
                                                       device=device)

    def update_metadata(self, mapping: LoRAMapping,
                        lora_index_to_id: list[Optional[int]], max_loras: int,
                        vocab_size: int, extra_vocab_size: int, **kwargs):

        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)

    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   lora_a_stacked: tuple[torch.Tensor,
                                         ...], scale: float, **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """
        x = x.view(-1, x.shape[-1])
        lora_shrink(
            x,
            lora_a_stacked,
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            scale,
        )

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            self._apply_bias(token_lora_indices, y, output_slices,
                             lora_bias_stacked)

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        num_tokens = x.size(1)  # first dimension is the num slices

        lora_expand(
            x,
            lora_b_stacked,
            y,
            *self.token_mapping_meta.meta_args(num_tokens),
            offset_start=offset_start,
            add_inputs=True,
        )

        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked, ),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """

        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            y = self._apply_bias(token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=torch.float32,
                device=x.device,
            )
        self.add_shrink(
            buffer,  # type: ignore
            x,
            lora_a_stacked,
            scale,
            **kwargs)
        self.add_expand(
            y,
            buffer,  # type: ignore
            lora_b_stacked,
            None,
            output_slices,
            add_inputs=True,
            **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        lora_shrink(x, [lora_a_stacked], buffer.unsqueeze(dim=0),
                    *self.prompt_mapping_meta.meta_args(x.size(0)), scale)

        lora_expand(buffer.unsqueeze(dim=0), [lora_b_stacked],
                    y,
                    *self.prompt_mapping_meta.meta_args(buffer.size(0)),
                    add_inputs=True)
        y = y.view_as(y_org)

@final
class AtmmWrapperGPU(PunicaWrapperBase):

    def __init__(self, max_num_batched_tokens: int, max_batches: int,
                 device: Union[torch.device, str], **kwargs):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches,
                                   device)

        self.max_loras = kwargs['max_loras']

        self.token_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                      max_num_batched_tokens,
                                                      device=device)

        # When cudagraph capture size is greater than max_num_seqs (max_batches,
        # here), V0 captures the graph as if max_num_seqs is set to
        # the capture size.
        # V1 doesn't have this problem and always respects max_num_seqs.

        max_num_prompts = (max_batches
                           if envs.VLLM_USE_V1 else max_num_batched_tokens)
        self.prompt_mapping_meta = LoRAKernelMeta.make(self.max_loras,
                                                       max_num_prompts,
                                                       device=device)
        import json
        try:
            with open("/data/miliang/vllm/vllm/lora/punica_wrapper/dict.json", "r") as f:
                self.shape_to_config = json.load(f)
                # logger.info(f"cjh shape_to_config: {self.shape_to_config}")
        except:
            self.shape_to_config = {}
        # 导入atmm_ops的dispatch_bgmv函数
        try:
            from atmm_ops import dispatch_bgmv as dispatch_sgmm
            self.dispatch_sgmm = dispatch_sgmm
        except ImportError:
            raise ImportError("Need to install atmm_ops to use AtmmWrapperGPU.")
        # 初始化内核参数默认值
        self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y = 32, 32, 32, 32, 32

    def update_metadata(self, mapping: LoRAMapping,
                        lora_index_to_id: list[Optional[int]], max_loras: int,
                        vocab_size: int, extra_vocab_size: int, **kwargs):
        # import traceback
        # stack = traceback.format_stack()
        # logger.info("update_metadata called from:\n" + "".join(stack))
        self.is_prefill = mapping.is_prefill
        self._update_base_metadata(mapping, lora_index_to_id, max_loras,
                                   vocab_size, extra_vocab_size)

        # Prepare cuda kernel metadata tensors
        self.token_mapping_meta.prepare_tensors(self.token_lora_indices)
        self.prompt_mapping_meta.prepare_tensors(self.sampler_indices)
    
    def add_shrink(self, y: torch.Tensor, x: torch.Tensor,
                   lora_a_stacked: tuple[torch.Tensor,
                                         ...], scale: float, 
                    a_start: Optional[torch.Tensor] = None,
                    a_len: Optional[torch.Tensor] = None,
                    a_loc: Optional[torch.Tensor] = None,
                    a_scaling: Optional[torch.Tensor] = None,
                    tmp_d: Optional[torch.Tensor] = None,
                    # rank_counts: Optional[torch.Tensor] = None,
                    **kwargs):
        """
        Performs GEMM  for multiple slices of lora_a.
            
        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        
        Args:
            y (torch.Tensor): Output tensors
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weights
            scale (float): Scaling factor for the operation
        """
        # logger.info("ATMM shrink!!!!!!")
        x = x.view(-1, x.shape[-1])
    
        # 获取基本参数
        # num_tokens, hidden_size = x.shape
        nslices = len(lora_a_stacked)
        
        # shrink操作的权重形状是 [num_loras, rank, hidden_size]
        layer_idx = kwargs.get("layer_idx", 0)  # 上游传入当前层索引
        lora_weights = lora_a_stacked[0][:, layer_idx, :, :].contiguous() 
        # logger.info(f"lora_weights.shape: {lora_weights.shape}")
        # logger.info(f"x shape: {x.shape}")
        # logger.info(f"y shape: {y.shape}")

        num_loras = lora_weights.size(0)
        rank = lora_weights.size(1)
        # 创建atmm_ops需要的参数
        # import time
        # torch.cuda.synchronize()
        # t0 = time.time()
        # a_start1 = torch.arange(0, num_loras * rank, step=rank, device=x.device, dtype=torch.long)
        # torch.cuda.synchronize()
        # t1 = time.time()
        # print(f"a_start elapsed: {(t1-t0)*1000:.3f} ms")

        # a_len1 = torch.full((num_loras,), rank, device=x.device, dtype=torch.long)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"a_len elapsed: {(t2-t1)*1000:.3f} ms")

        # a_loc1 = torch.arange(0, num_loras * rank, device=x.device, dtype=torch.long)
        # torch.cuda.synchronize()
        # t3 = time.time()
        # print(f"a_loc elapsed: {(t3-t2)*1000:.3f} ms")

        # a_scaling1 = torch.full((num_loras,), scale, device=x.device, dtype=x.dtype)
        # torch.cuda.synchronize()
        # t4 = time.time()
        # print(f"a_scaling elapsed: {(t4-t3)*1000:.3f} ms")
        # logger.info(f"cjh x.dtype: {x.dtype}")
        # 创建临时缓冲区
        # N0 = 32 * 4096
        # tmp_d1 = torch.zeros(N0 * 60, dtype=torch.int8, device=x.device)
        # t5 = time.time()
        # print(f"tmp_d elapsed: {(t5-t4)*1000:.3f} ms")

        batches = 1 if self.batch_size < 0 else self.batch_size
        rank_counts = torch.full((batches,), rank, device=x.device, dtype=torch.long)
        # logger.info(f"cjh batches: {self.batch_size, batches}")
        # logger.info(f"cjh self._seq_lengths: {self._seq_lengths.shape}")
        # t6 = time.time()
        # print(f"rank_counts elapsed: {(t6-t5)*1000:.3f} ms")

        shape =  f"{int(batches)}, {int(x.size(0))}, {int(x.size(1))}, {int(rank)}"
        if shape in self.shape_to_config:
            (self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y) = self.shape_to_config[shape]
            # logger.info(f"111cjh shape_to_config[shape]: {self.shape_to_config[shape]}") 
        else:
            self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y = 32, 32, 32, 32, 32
        # t7 = time.time()
        # print(f"shape elapsed: {(t7-t6)*1000:.3f} ms")
        for slice_id in range(nslices):
            self.dispatch_sgmm(
                y[slice_id],  # 输出
                x,  # 输入
                lora_a_stacked[slice_id][:, layer_idx, :, :].contiguous() ,  # 权重
                a_start, a_len, a_loc,
                0, a_scaling,
                self._seq_lengths, rank_counts, self._lora_indices_per_batch, self._seq_start_locs, tmp_d,
                batches, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.wp_y
            )
        # t8 = time.time()
        # print(f"dispatch_sgmm elapsed: {(t8-t7)*1000:.3f} ms")

    def add_expand(self,
                   y: torch.Tensor,
                   x: torch.Tensor,
                   lora_b_stacked: tuple[torch.Tensor, ...],
                   lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                   output_slices: tuple[int, ...],
                   offset_start: int = 0,
                   a_start: Optional[torch.Tensor] = None,
                   a_len: Optional[torch.Tensor] = None,
                   a_loc: Optional[torch.Tensor] = None,
                   a_scaling: Optional[torch.Tensor] = None,
                   tmp_d: Optional[torch.Tensor] = None,
                   # rank_counts: Optional[torch.Tensor] = None,
                   add_inputs=True,
                   **kwargs) -> None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.
      
        Semantics:
            for i in range(len(lora_b_stacked)):
                slice = output_slices[i]
                y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i] + 
                    lora_bias_stacked[i] 
                offset += slice
            
        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensors
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): 
                bias's weight
            output_slices (tuple[int, ...]): Every slice's size
            add_inputs (bool): Defaults to True.
        """
        # logger.info("ATMM expand!!!!!!")
        y_org = y
        y = y.view(-1, y.shape[-1])
        if lora_bias_stacked is not None:
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                            y.size(0))
            self._apply_bias(token_lora_indices, y, output_slices,
                            lora_bias_stacked)

        assert x.ndim == 3
        assert x.size(0) == len(output_slices)
        # num_tokens = x.size(1)  # first dimension is the num slices

        # 获取基本参数
        # num_tokens, hidden_size = x.shape
        nslices = len(lora_b_stacked)
        
        # expand操作的权重形状是 [num_loras, hidden_size, rank]
        layer_idx = kwargs.get("layer_idx", 0)  # 上游传入当前层索引
        lora_weights = lora_b_stacked[0][:, layer_idx, :, :].contiguous() 
        # logger.info(f"lora_weights.shape: {lora_weights.shape}")
        # logger.info(f"x shape: {x.shape}")
        # logger.info(f"y shape: {y.shape}")
        # logger.info(f"output_slices: {output_slices}")
        num_loras = lora_weights.size(0)
        rank = lora_weights.size(2)
        # 创建atmm_ops需要的参数
        # a_start1 = torch.arange(0, num_loras * rank, step=rank, device=x.device, dtype=torch.long)
        # a_len1 = torch.full((num_loras,), rank, device=x.device, dtype=torch.long)
        # a_loc1 = torch.arange(0, num_loras * rank, device=x.device, dtype=torch.long)
        # a_scaling1 = torch.ones(num_loras, device=x.device, dtype=x.dtype)
        
        # 创建临时缓冲区
        # N0 = 32 * 4096
        # tmp_d1 = torch.zeros(N0 * 60, dtype=torch.int8, device=x.device)
        
        batches = 1 if self.batch_size < 0 else self.batch_size
        rank_counts = torch.full((batches,), rank, device=x.device, dtype=torch.long)
        shape = (batches, x.shape[0], x.shape[1], rank)
        if shape in self.shape_to_config:
            (self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y) = self.shape_to_config[shape]
            logger.info(f"111cjh shape_to_config[shape]: {self.shape_to_config[shape]}") 
        else:
            self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y = 32, 32, 32, 32, 32
        for slice_id in range(nslices):
            offset = 0
            slice = output_slices[slice_id]
            self.dispatch_sgmm(
                y[:, offset:offset+slice].contiguous(),  # 输出
                x[slice_id],  # 输入
                lora_b_stacked[slice_id][:, layer_idx, :, :].transpose(1, 2).contiguous(),  # 权重
                a_start, a_len, a_loc,
                0, a_scaling,
                self._seq_lengths, rank_counts, self._lora_indices_per_batch, self._seq_start_locs, tmp_d,
                batches, self.tb_x, self.tb_y, self.tb_z, self.wp_x, self.wp_y, self.wp_y
            )
            offset += slice

        y = y.view_as(y_org)

    def add_lora_embedding(self,
                           y: torch.Tensor,
                           x: torch.Tensor,
                           lora_b_stacked: torch.Tensor,
                           add_inputs: bool = True,
                           **kwargs) -> None:
        """
        Applies lora  specifically for VocabParallelEmbeddingWithLoRA.

        Semantics:
            y += x @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            add_inputs (bool): Default to True.
        """

        lora_expand(
            x.unsqueeze(dim=0),
            (lora_b_stacked, ),
            y,
            *self.token_mapping_meta.meta_args(x.size(0)),
            offset_start=0,
            add_inputs=add_inputs,
        )

    def add_lora_linear(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: tuple[torch.Tensor, ...],
                        lora_b_stacked: tuple[torch.Tensor, ...],
                        lora_bias_stacked: Optional[tuple[torch.Tensor, ...]],
                        scale: float,
                        output_slices: tuple[int, ...],
                        a_start: Optional[torch.Tensor] = None,
                        a_len: Optional[torch.Tensor] = None,
                        a_loc: Optional[torch.Tensor] = None,
                        a_scaling: Optional[torch.Tensor] = None,
                        tmp_d: Optional[torch.Tensor] = None,
                        # rank_counts: Optional[torch.Tensor] = None,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applicable to linear-related lora. 

        Semantics:
            for i in range(len(lora_a_stacked)):
                y[i] += (
                    x[i].unsqueeze(0)
                    @ lora_a_stacked[indices[i], layer_idx, :, :]
                    @ lora_b_stacked[indices[i], layer_idx, :, :]
                    * scale
                    ).squeeze(0)+lora_bias_stacked[i]

        Args:
            y (torch.Tensor): Output tensor. Will be changed in-place.
            x (torch.Tensor): Input tensor
            lora_a_stacked (tuple[torch.Tensor, ...]): lora_a's weight.
            lora_b_stacked (tuple[torch.Tensor, ...]): lora_b's weight.
            lora_bias_stacked (Optional[tuple[torch.Tensor, ...]]): lora's bias.
            scale (float): Scaling factor.
            output_slices (tuple[int, ...]): Every slice's size.
            buffer (Optional[torch.Tensor]): Defaults to None.
        """
        # logger.info("add lora linear!!!!!!")
        # import traceback
        # stack = traceback.format_stack()
        # logger.info("add_lora_linear called from:\n" + "".join(stack))
        assert len(lora_a_stacked) == len(lora_b_stacked) == len(output_slices)
        if lora_bias_stacked is not None:
            assert len(lora_bias_stacked) == len(output_slices)
            token_lora_indices = torch.narrow(self._token_lora_indices, 0, 0,
                                              y.size(0))
            y = self._apply_bias(token_lora_indices, y, output_slices,
                                 lora_bias_stacked)

        if buffer is None:
            r = lora_b_stacked[0].size(-1)
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros(  # type: ignore
                (len(output_slices), x.size(0), r),
                dtype=x.dtype,
                device=x.device,
            )
        # 判断a_scaling是否为None，如果为None，则不进行shrink和expand
        if a_scaling is not None:
            self.add_shrink(
                buffer,  # type: ignore
                x,
                lora_a_stacked,
                scale,
                a_start=a_start, a_len=a_len, a_loc=a_loc, a_scaling=a_scaling, tmp_d=tmp_d,
                **kwargs)
            self.add_expand(
                y,
                buffer,  # type: ignore
                lora_b_stacked,
                None,
                output_slices,
                a_start=a_start, a_len=a_len, a_loc=a_loc, a_scaling=a_scaling, tmp_d=tmp_d,
                add_inputs=True,
                **kwargs)

    def add_lora_logits(self,
                        y: torch.Tensor,
                        x: torch.Tensor,
                        lora_a_stacked: torch.Tensor,
                        lora_b_stacked: torch.Tensor,
                        scale,
                        *,
                        buffer: Optional[torch.Tensor] = None,
                        **kwargs) -> None:
        """
        Applies lora  specifically for LogitsProcessorWithLoRA.
        
        Semantics:
            buffer = (x @ lora_a_stacked) * scale
            y += buffer @ lora_b_stacked

        Args:
            y (torch.Tensor): Output tensor.
            x (torch.Tensor): Input tensor.
            lora_a_stacked (torch.Tensor): lora_a's weights.
            lora_b_stacked (torch.Tensor): lora_b's weights.
            scale (float): Scaling factor.
            buffer (Optional[torch.Tensor]): Default to None.
        """
        y_org = y
        y = y.view(-1, y.shape[-1])
        x = x.view(-1, x.shape[-1])
        r = lora_b_stacked.size(-1)
        if buffer is None:
            # We set the buffer to be float32 by default, refer to:
            # https://github.com/triton-lang/triton/issues/1387
            buffer = torch.zeros((x.size(0), r),
                                 dtype=torch.float32,
                                 device=x.device)

        lora_shrink(x, [lora_a_stacked], buffer.unsqueeze(dim=0),
                    *self.prompt_mapping_meta.meta_args(x.size(0)), scale)

        lora_expand(buffer.unsqueeze(dim=0), [lora_b_stacked],
                    y,
                    *self.prompt_mapping_meta.meta_args(buffer.size(0)),
                    add_inputs=True)
        y = y.view_as(y_org)
