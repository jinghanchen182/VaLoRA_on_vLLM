# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from threading import Lock

import pytest
import torch

import vllm.lora.ops.torch_ops as torch_ops
import vllm.lora.ops.triton_ops as triton_ops
from vllm.lora.ops.triton_ops import LoRAKernelMeta
from vllm.lora.ops.triton_ops.utils import _LORA_A_PTR_DICT, _LORA_B_PTR_DICT
from vllm.platforms import current_platform
from atmm_ops import dispatch_bgmv as dispatch_sgmm

from .utils import PunicaTensors, assert_close, generate_data_for_nslices


@pytest.fixture(autouse=True)
def reset_device(reset_default_device):
    pass


# Utility shrink and expand operations used as reference implementations.
def sgmv_shrink_for_nslices(
        nslices: int, inputs_tensor: torch.Tensor,
        lora_weights_lst: list[torch.Tensor], out_tensor: torch.Tensor,
        b_seq_start_loc: torch.Tensor, seq_len_tensor: torch.Tensor,
        prompt_lora_mapping: torch.Tensor, batches: int, max_seq_length: int,
        num_tokens: int, scaling: float):
    """
    Wrapper around torch_ops.sgmv_shrink that handles any nslices.
    """
    for index in range(nslices):
        torch_ops.sgmv_shrink(
            inputs_tensor,
            lora_weights_lst[index],
            out_tensor[index],
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            scaling,
        )


def sgmv_expand_for_nslices(nslices: int, hidden_size: int,
                            inputs_tensor: torch.Tensor,
                            lora_weights_lst: list[torch.Tensor],
                            out_tensor: torch.Tensor,
                            b_seq_start_loc: torch.Tensor,
                            seq_len_tensor: torch.Tensor,
                            prompt_lora_mapping: torch.Tensor, batches: int,
                            max_seq_length: int, num_tokens: int,
                            add_inputs: bool) -> None:
    """
    Wrapper around torch_ops.sgmv_expand that handles any nslices.
    """
    if nslices == 1:
        # Verify the torch's sgmv_expand op
        torch_ops.sgmv_expand(
            inputs_tensor[0],
            lora_weights_lst[0],
            out_tensor,
            b_seq_start_loc,
            seq_len_tensor,
            prompt_lora_mapping,
            batches,
            max_seq_length,
            num_tokens,
            add_inputs=add_inputs,
        )
    else:
        slice_offset = 0
        for index in range(nslices):
            lora_weights = lora_weights_lst[index]
            torch_ops.sgmv_expand_slice(
                inputs_tensor[index],
                lora_weights,
                out_tensor,
                b_seq_start_loc,
                seq_len_tensor,
                prompt_lora_mapping,
                batches,
                max_seq_length,
                num_tokens,
                slice_offset,
                hidden_size,
                add_inputs=add_inputs,
            )
            slice_offset += hidden_size


_dict_lock = Lock()


def check_lora_shrink_kernel(batches: int, num_loras: int, rank: int,
                             hidden_size: int, nslices: int,
                             dtype: torch.dtype, device: str, seq_length: int,
                             scaling: float):
    """
    Compare outputs of torch_ops.sgmv_shrink and triton_ops.lora_shrink
    kernels.
    """
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "shrink",
        device,
    )
    max_seq_length, token_nums = data.meta()

    # Setup metadata information for SGMV and reference kernels
    sgmv_meta_args = (data.b_seq_start_loc, data.seq_len_tensor,
                      data.prompt_lora_mapping, batches, max_seq_length,
                      token_nums)

    # Setup metadata information for the LoRA kernel.
    lora_meta = LoRAKernelMeta.make(max_loras=num_loras,
                                    max_num_tokens=token_nums,
                                    device='cuda')
    lora_meta.prepare_tensors(data.token_lora_mapping)
    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    # Preventing cache error pointer.
    with _dict_lock:
        _LORA_A_PTR_DICT.clear()
        
        if isinstance(data.lora_weights, list):
            lora_weights = data.lora_weights[0].contiguous() 
        else:
            lora_weights = data.lora_weights.contiguous()
        
        
        for slice_id in range(nslices):
            # 将具有相同LoRA ID的连续批次合并成组
            # groups = []
            # cur_id = None; cur_start = 0; cur_len = 0
            # for b in range(batches):
            #     lora_id = int(data.prompt_lora_mapping[b]) # 当前批次使用的LoRA ID
            #     start_id = int(data.b_seq_start_loc[b])    # 当前批次在序列中的起始位置
            #     seq_len = int(data.seq_len_tensor[b])      # 当前批次的序列长度
            #     if lora_id != cur_id:
            #         if cur_id is not None:
            #             groups.append((cur_id, cur_start, cur_len))
            #         cur_id, cur_start, cur_len = lora_id, start_id, seq_len
            #     else: # 如果LoRA ID相同，累加序列长度到当前组
            #         cur_len += seq_len
            # if cur_id is not None:
            #     groups.append((cur_id, cur_start, cur_len))
            
            # groups = []
            # for b in range(batches):
            #     lora_id = int(data.prompt_lora_mapping[b])  # 当前批次使用的LoRA ID
            #     start_id = int(data.b_seq_start_loc[b])     # 当前批次在序列中的起始位置
            #     seq_len = int(data.seq_len_tensor[b])       # 当前批次的序列长度
            #     current_start = start_id
            #     remaining = seq_len
            #     while remaining > 0:
            #         batch_len = min(64, remaining)
            #         groups.append((lora_id, current_start, batch_len))
            #         current_start += batch_len
            #         remaining -= batch_len

            # num_problems = len(groups)
        
            N0 = 32 * 4096
            tmp_d = torch.zeros(N0 * 60, dtype=torch.int8, device="cuda")
            # output_counts = torch.zeros(N0, dtype=torch.long, device="cuda")
            # rank_counts = torch.zeros(N0, dtype=torch.long, device="cuda")
            # lora_ids = torch.zeros(N0, dtype=torch.long, device="cuda")
            # start_ids = torch.zeros(N0, dtype=torch.long, device="cuda")
            # output_counts[0], output_counts[1] = 128, 128
            # rank_counts[0], rank_counts[1] = 16, 16
            # start_ids[0] = 0
            # start_ids[1] = 128

            # a_len = torch.tensor([4096 * 4] * num_adapters, dtype=torch.long, device="cuda")
            # a_start = torch.zeros_like(a_len)
            # a_start[1:] = torch.cumsum(a_len[:-1], dim=0)
            # a_loc = torch.arange(4096 * 4 * num_adapters, dtype=torch.long, device="cuda")
            # a_scaling = torch.tensor([1] * num_adapters, dtype=torch.float16, device="cuda")
            # lora_ids      = torch.tensor([g[0] for g in groups], device=device, dtype=torch.long)
            # start_ids     = torch.tensor([g[1] for g in groups], device=device, dtype=torch.long)
            # output_counts = torch.tensor([g[2] for g in groups], device=device, dtype=torch.long)  # m
            rank_counts   = torch.full((batches,), rank, device=device, dtype=torch.long)     # k

            a_start  = torch.arange(0, num_loras*rank, step=rank, device=device, dtype=torch.long)   # [num_loras]
            a_len    = torch.full((num_loras,), rank, device=device, dtype=torch.long)               # [num_loras]
            a_loc    = torch.arange(0, num_loras*rank, device=device, dtype=torch.long)              # [num_loras*rank]
            a_scaling = torch.full((num_loras,), scaling, device=device, dtype=dtype)  # [num_loras]
            
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            print(lora_weights.shape)
            # lora_weights = torch.zeros(num_loras*rank, 32, 128, device=device, dtype=dtype)
            # inputs_tensor = torch.zeros(512, 4096, device=device, dtype=dtype)
            # import pdb; pdb.set_trace()
            lora_weights = torch.randn(num_loras, rank, hidden_size, device=device, dtype=dtype).half()
            dispatch_sgmm(
                out_tensor[slice_id],  # 输出 [num_tokens, rank]
                data.inputs_tensor,  # 输入 [num_tokens, hidden_size]
                lora_weights, # 权重 [lora_nums, rank, hidden_size]
                a_start, a_len, a_loc,
                # data.prompt_lora_mapping, 
                0, a_scaling,
                data.seq_len_tensor, rank_counts, data.prompt_lora_mapping, data.b_seq_start_loc, tmp_d,
                batches, 32, 32, 32, 32, 32, 32)  
            # import pdb; pdb.set_trace()
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            print(f"dispatch_sgmm elapsed: {elapsed_ms:.3f} ms")
            # print(out_tensor[slice_id])
            # nan_rows = torch.any(torch.isnan(out_tensor[0]), dim=1)
            # num_non_nan_rows = out_tensor[0].shape[0] - torch.sum(nan_rows)
            # print(f"\nNumber of non-NaN rows: {num_non_nan_rows.item()}")
                # print(f"lora_weight:{data.lora_weights[slice_id].shape}")

        
    # Reference
    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        [lora_weights],
        ref_out_tensor,
        *sgmv_meta_args,
        scaling,
    )
    # import pdb; pdb.set_trace()
    # print(ref_out_tensor)
    assert_close(out_tensor, ref_out_tensor)



def check_lora_expand_kernel(batches: int, num_loras: int, rank: int,
                             hidden_size: int, nslices: int,
                             dtype: torch.dtype, device: str, seq_length: int,
                             add_inputs: bool):
    """
    Compare outputs of torch_ops.sgmv_expand and triton_ops.lora_expand
    kernels.
    """
    data: PunicaTensors = generate_data_for_nslices(
        batches,
        hidden_size,
        num_loras,
        rank,
        seq_length,
        nslices,
        dtype,
        "expand",
        device,
    )

    max_seq_length, token_nums = data.meta()

    # Setup metadata information for SGMV and reference kernels
    sgmv_meta_args = (data.b_seq_start_loc, data.seq_len_tensor,
                      data.prompt_lora_mapping, batches, max_seq_length,
                      token_nums)

    # Setup metadata information for the LoRA kernel.
    lora_meta = LoRAKernelMeta.make(max_loras=num_loras,
                                    max_num_tokens=token_nums,
                                    device='cuda')
    lora_meta.prepare_tensors(data.token_lora_mapping)

    # Setup output tensors
    ref_out_tensor = data.ref_out_tensor
    out_tensor = data.our_out_tensor.clone()

    with _dict_lock:
        # lora_expand kernel
        _LORA_B_PTR_DICT.clear()

        if isinstance(data.lora_weights, list):
            lora_weights = data.lora_weights[0].contiguous() 
        else:
            lora_weights = data.lora_weights.contiguous()
        
        for slice_id in range(nslices):
            N0 = 32 * 4096
            tmp_d = torch.zeros(N0 * 60, dtype=torch.int8, device="cuda")
            rank_counts   = torch.full((batches,), rank, device=device, dtype=torch.long)     # k

            a_start  = torch.arange(0, num_loras*rank, step=rank, device=device, dtype=torch.long)   # [num_loras]
            a_len    = torch.full((num_loras,), rank, device=device, dtype=torch.long)               # [num_loras]
            a_loc    = torch.arange(0, num_loras*rank, device=device, dtype=torch.long)              # [num_loras*hidden_size]
            a_scaling = torch.full((num_loras,), 1, device=device, dtype=dtype)  # [num_loras]
            torch.cuda.synchronize()
            # import pdb; pdb.set_trace()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            lora_weights = torch.randn(num_loras, hidden_size, rank, device=device, dtype=dtype)
            print(f"lora_weights.shape: {lora_weights.shape}")
            print(f"x shape: {data.inputs_tensor[slice_id].shape}")
            print(f"y shape: {out_tensor.shape}")
            dispatch_sgmm(
                out_tensor,  # 输出 [num_tokens, hidden_size * nslices]
                data.inputs_tensor[slice_id],  # 输入 [num_tokens, max_rank]
                lora_weights.transpose(1, 2).contiguous(), # 权重 [lora_nums, hidden_size, max_rank]
                a_start, a_len, a_loc,
                # data.prompt_lora_mapping, 
                0, a_scaling,
                data.seq_len_tensor, rank_counts, data.prompt_lora_mapping, data.b_seq_start_loc, tmp_d,
                batches, 32, 32, 32, 32, 32, 32)  
            end_event.record()
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            print(f"dispatch_sgmm elapsed: {elapsed_ms:.3f} ms")
            # print(out_tensor)

    # Reference
    sgmv_expand_for_nslices(nslices,
                            hidden_size,
                            data.inputs_tensor,
                            [lora_weights],
                            ref_out_tensor,
                            *sgmv_meta_args,
                            add_inputs=add_inputs)
    # import pdb; pdb.set_trace()
    assert_close(out_tensor, ref_out_tensor)


# Tests
# We test the punica kernels along 2 verticals mainly.
# 1. Variations in hidden_dim size
# 2. Variations in all other parameters like (batch_size, max_rank, num_loras
#  etc.)

# We have collected the hidden_sizes included in the LoRA models
# currently supported by vLLM. It tests whether the corresponding Triton
# kernel can run normally when tensor parallelism is set to
# [1, 2, 4, 8, 16, 32, 64].
HIDDEN_SIZES = [
    128,
    256,
    512,
    896,
    1024,
    1152,
    1216,
    1280,
    1536,
    1664,
    2048,
    2240,
    2304,
    2368,
    2432,
    2560,
    2752,
    3072,
    3328,
    3456,
    3584,
    3712,
    4096,
    4480,
    4608,
    4736,
    4864,
    5120,
    5504,
    5632,
    5888,
    6144,
    6400,
    6848,
    6912,
    7168,
    7424,
    8192,
    8960,
    9216,
    9472,
    10240,
    11008,
    11264,
    13824,
    14336,
    14784,
    14848,
    15360,
    18944,
    22016,
    22528,
    24576,
    27392,
    27648,
    29568,
    29696,
    32000,
    32256,
    32512,
    32768,
    33024,
    36864,
    43264,
    49152,
    49408,
    60544,
    60672,
    64000,
    64256,
    102400,
    102656,
    128000,
    128256,
]
#The size of TP
divisibility = [1, 2, 8, 16, 64]

all_hidden_size = []
for div in divisibility:
    for hidden_size in HIDDEN_SIZES:
        all_hidden_size.append(hidden_size // div)

HIDDEN_SIZES = list(set(all_hidden_size))

# Test params that focuses on hidden_size variation.
hs_test_params = {
    "hidden_sizes": [4096],
    "batches": [1],
    "num_loras": [1],
    "max_ranks": [16],
}

# General tests params that tests for variations in all dimensions
# except hidden_size.
test_params = {
    "hidden_sizes": [4096], # 修改过，原来是2049
    "batches": [8, 16, 32, 64],
    "num_loras": [1, 2, 4, 8, 16, 32, 64],
    "max_ranks": [16],
}

DTYPES = [torch.float16]
DEVICES = [f"cuda:{0}"]
SEED = [0]


@pytest.mark.parametrize("batches", test_params['batches'])
@pytest.mark.parametrize("num_loras", test_params['num_loras'])
@pytest.mark.parametrize("rank", test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [1])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["expand"])
def test_kernels(
    batches: int,
    num_loras: int,
    rank: int,
    hidden_size: int,
    nslices: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
    op_type: str,
):
    """
    Tests LoRA kernels.
    """
    torch.set_default_device(device)
    current_platform.seed_everything(seed)

    if op_type == "shrink":
        check_lora_shrink_kernel(batches=batches,
                                 num_loras=num_loras,
                                 rank=rank,
                                 hidden_size=hidden_size,
                                 nslices=nslices,
                                 dtype=dtype,
                                 device=device,
                                 seq_length=2048,
                                 scaling=1)
    else:
        check_lora_expand_kernel(batches=batches,
                                 num_loras=num_loras,
                                 rank=rank,
                                 hidden_size=hidden_size,
                                 nslices=nslices,
                                 dtype=dtype,
                                 device=device,
                                 seq_length=2048,
                                 add_inputs=True)


# @pytest.mark.parametrize("batches", hs_test_params['batches'])
# @pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
# @pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
# @pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
# @pytest.mark.parametrize("nslices", [1])
# @pytest.mark.parametrize("dtype", DTYPES)
# @pytest.mark.parametrize("device", DEVICES)
# @pytest.mark.parametrize("seed", SEED)
# @pytest.mark.parametrize("op_type", ["shrink"])
# def test_kernels_hidden_size(
#     batches: int,
#     num_loras: int,
#     rank: int,
#     hidden_size: int,
#     nslices: int,
#     dtype: torch.dtype,
#     device: str,
#     seed: int,
#     op_type: str,
# ):
#     """
#     Tests SGMV and LoRA kernels.
#     """
#     torch.set_default_device(device)
#     current_platform.seed_everything(seed)

#     if op_type == "shrink":
#         check_lora_shrink_kernel(batches=batches,
#                                  num_loras=num_loras,
#                                  rank=rank,
#                                  hidden_size=hidden_size,
#                                  nslices=nslices,
#                                  dtype=dtype,
#                                  device=device,
#                                  seq_length=64,
#                                  scaling=1)
#     else:
#         check_lora_expand_kernel(batches=batches,
#                                  num_loras=num_loras,
#                                  rank=rank,
#                                  hidden_size=hidden_size,
#                                  nslices=nslices,
#                                  dtype=dtype,
#                                  device=device,
#                                  seq_length=128,
#                                  add_inputs=True)
