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

    ref_out_tensor = data.ref_out_tensor.half()
    out_tensor = data.our_out_tensor.clone().half()

    # Preventing cache error pointer.
    with _dict_lock:
        # lora_shrink kernel
        _LORA_A_PTR_DICT.clear()

        lora_weights = data.lora_weights if isinstance(data.lora_weights, list) else [data.lora_weights]
        
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
            
            groups = []
            for b in range(batches):
                lora_id = int(data.prompt_lora_mapping[b])  # 当前批次使用的LoRA ID
                start_id = int(data.b_seq_start_loc[b])     # 当前批次在序列中的起始位置
                seq_len = int(data.seq_len_tensor[b])       # 当前批次的序列长度
                current_start = start_id
                remaining = seq_len
                while remaining > 0:
                    batch_len = min(32, remaining)
                    groups.append((lora_id, current_start, batch_len))
                    current_start += batch_len
                    remaining -= batch_len

            num_problems = len(groups)
            lora_ids      = torch.tensor([g[0] for g in groups], device=device, dtype=torch.long)
            start_ids     = torch.tensor([g[1] for g in groups], device=device, dtype=torch.long)
            output_counts = torch.tensor([g[2] for g in groups], device=device, dtype=torch.long)  # m
            rank_counts   = torch.full((num_problems,), rank, device=device, dtype=torch.long)     # k

            a_start  = torch.arange(0, num_loras*rank, step=rank, device=device, dtype=torch.long)   # [num_loras]
            a_len    = torch.full((num_loras,), rank, device=device, dtype=torch.long)               # [num_loras]
            a_loc    = torch.arange(0, num_loras*rank, device=device, dtype=torch.long)              # [num_loras*rank]
            a_scaling = torch.full((num_loras,), scaling, device=device, dtype=dtype)  # [num_loras]

            total_tokens = data.inputs_tensor.shape[0]  # 实际的token数
            N = max(32, total_tokens) * 4096  # 确保缓冲区足够大
            tmp_d = torch.zeros(N * 60, dtype=torch.int8, device="cuda")
            

            # print(f"=== Python调试信息 ===")
            print(f"num_problems: {num_problems}")
            print(f"groups: {groups}")
            print(f"output_counts: {output_counts}")
            print(f"rank_counts: {rank_counts}")
            print(f"lora_ids: {lora_ids}")
            print(f"start_ids: {start_ids}")
            print(f"num_loras: {num_loras}")
            # print(f"a_loc shape: {a_loc.shape}, a_start shape: {a_start.shape}, a_len shape: {a_len.shape}")
            # print(f"total_tokens: {total_tokens}")
            
            # 调用 atmm 内核
            print(a_loc)
            print(a_len)
            print(a_scaling)
            print(f"input shape:{data.inputs_tensor.shape}")
            print(f"lora_weight:{lora_weights[slice_id].shape}")
            print(f"output_shape: {out_tensor[slice_id].shape}")
            dispatch_sgmm(
                out_tensor[slice_id],  # 输出 shape=[batchsize, lora_rank] [num_tokens, rank]
                data.inputs_tensor,  # 输入 shape=[batchsize, hidden_size] [num_tokens, hidden_size]
                lora_weights[slice_id],   # loraA shape=[lora_nums, hidden_size, rank]
                a_start, a_len, a_loc,
                # data.prompt_lora_mapping, 
                0, a_scaling,
                output_counts, rank_counts, lora_ids, start_ids, tmp_d,
                num_problems,32, 32, 32, 32, 32, 32)  
            torch.cuda.synchronize() 
            print(out_tensor[slice_id])
            nan_rows = torch.any(torch.isnan(out_tensor[slice_id]), dim=1)
            num_non_nan_rows = out_tensor[slice_id].shape[0] - torch.sum(nan_rows)
            print(f"\nNumber of non-NaN rows: {num_non_nan_rows.item()}")
            # print(f"lora_weight:{data.lora_weights[slice_id].shape}")

        
    # Reference
    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        scaling,
    )
    import pdb; pdb.set_trace()
    # print(ref_out_tensor)
    assert_close(out_tensor, ref_out_tensor)


def check_lora_shrink_kernel_v1(batches: int, num_loras: int, rank: int,
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

    ref_out_tensor = data.ref_out_tensor.half()
    out_tensor = data.our_out_tensor.clone().half()

    # Preventing cache error pointer.
    with _dict_lock:
        # lora_shrink kernel
        _LORA_A_PTR_DICT.clear()

        lora_weights = data.lora_weights if isinstance(data.lora_weights, list) else [data.lora_weights]
        
        for slice_id in range(nslices):
            groups = []
            cur_id = None; cur_start = 0; cur_len = 0
            for b in range(batches):
                lora_id = int(data.prompt_lora_mapping[b])
                start_id = int(data.b_seq_start_loc[b])
                seq_len = int(data.seq_len_tensor[b])
                if lora_id != cur_id:
                    if cur_id is not None:
                        groups.append((cur_id, cur_start, cur_len))
                    cur_id, cur_start, cur_len = lora_id, start_id, seq_len
                else:
                    cur_len += seq_len
            if cur_id is not None:
                groups.append((cur_id, cur_start, cur_len))
            
            # 公共数组定义
            a_start  = torch.arange(0, num_loras*rank, step=rank, device=device, dtype=torch.long)   # [num_loras]
            a_len    = torch.full((num_loras,), rank, device=device, dtype=torch.long)               # [num_loras]
            a_loc    = torch.arange(0, num_loras*rank, device=device, dtype=torch.long)              # [num_loras*rank]
            a_scaling = torch.full((num_loras,), scaling, device=device, dtype=dtype)  # [num_loras]
            
            # workspace_multiplier = max(data.inputs_tensor.shape[0], 32)  # 基础workspace大小
            # N = workspace_multiplier * 4096  
            # tmp_d = torch.zeros(N * 60, dtype=torch.int8, device="cuda")
            tmp_d = torch.zeros(32 * 4096 * 60, dtype=torch.int8, device="cuda")
            
            max_batch_size = 32
            original_groups = groups[:]
            if any(g[2] > max_batch_size for g in original_groups):
                print(f"Large batch detected, splitting at Python level")
                # 对于大batch，我们分别调用dispatch_sgmm
                for lora_id, start_id, total_len in original_groups:
                    remaining_len = total_len
                    current_start = start_id
                    while remaining_len > 0:
                        batch_len = min(remaining_len, max_batch_size)
                        print(f"Processing sub-batch: lora_id={lora_id}, start={current_start}, len={batch_len}")
                        
                        # 关键修复：创建实际的tensor slice而不是依赖指针偏移
                        sub_groups = [(lora_id, 0, batch_len)]  # 注意：start改为0，因为我们传递slice
                        sub_num_problems = len(sub_groups)
                        sub_lora_ids      = torch.tensor([g[0] for g in sub_groups], device=device, dtype=torch.long)
                        sub_start_ids     = torch.tensor([g[1] for g in sub_groups], device=device, dtype=torch.long)  # 总是0
                        sub_output_counts = torch.tensor([g[2] for g in sub_groups], device=device, dtype=torch.long)
                        sub_rank_counts   = torch.full((sub_num_problems,), rank, device=device, dtype=torch.long)
                        
                        # 创建tensor slice并确保连续性
                        input_slice = data.inputs_tensor[current_start:current_start+batch_len].contiguous()
                        output_slice = out_tensor[slice_id][current_start:current_start+batch_len].contiguous()
                        
                        print(f"  input_slice shape: {input_slice.shape}")
                        print(f"  output_slice shape: {output_slice.shape}")
                        print(f"  input_slice contiguous: {input_slice.is_contiguous()}")
                        print(f"  output_slice contiguous: {output_slice.is_contiguous()}")
                        print(f"  sub_start_ids: {sub_start_ids}")
                        
                        # 对比测试：使用PyTorch标准GEMM验证数据正确性
                        weight_2d = data.lora_weights[slice_id][0].float()  # [16, 4096]
                        input_float = input_slice.float()  # [32, 4096]
                        expected_output = torch.mm(input_float, weight_2d.T)  # [32, 16]
                        print(f"  PyTorch GEMM result[0:4, 0:4]:\n{expected_output[0:4, 0:4]}")
                        print(f"  PyTorch GEMM has nan: {torch.isnan(expected_output).any()}")
                        
                        # import pdb; pdb.set_trace()
                        # 单独调用dispatch_sgmm，使用slice
                        dispatch_sgmm(
                            output_slice,      # 使用slice而不是完整tensor
                            input_slice,       # 使用slice而不是完整tensor
                            data.lora_weights[slice_id],
                            a_start, a_len, a_loc,
                            0, a_scaling,
                            sub_output_counts, sub_rank_counts, sub_lora_ids, sub_start_ids, tmp_d,
                            sub_num_problems, 32, 32, 32, 32, 32, 32)
                        # 添加同步，确保 CUDA 端计算完成
                        torch.cuda.synchronize()
                        
                        # 检查这个slice的计算结果
                        print(f"  After GEMM - output_slice[0:4, 0:4]:\n{output_slice[0:4, 0:4]}")
                        print(f"  output_slice has nan: {torch.isnan(output_slice).any()}")
                        
                        # 将结果复制回原tensor（因为我们使用了contiguous()可能创建了副本）
                        out_tensor[slice_id][current_start:current_start+batch_len] = output_slice
                        
                        current_start += batch_len
                        remaining_len -= batch_len
                # Skip the main dispatch_sgmm call since we already processed everything
                out_tensor = out_tensor.contiguous()
                print("Large batch processing completed")
            
            else:
                print(f"使用原始groups（无需分解）: {groups}")
                
                # 正常的小batch处理
                num_problems = len(groups)
                lora_ids      = torch.tensor([g[0] for g in groups], device=device, dtype=torch.long)
                start_ids     = torch.tensor([g[1] for g in groups], device=device, dtype=torch.long)
                output_counts = torch.tensor([g[2] for g in groups], device=device, dtype=torch.long)  # m
                rank_counts   = torch.full((num_problems,), rank, device=device, dtype=torch.long)     # k
                
                print(f"=== Python调试信息 ===")
                print(f"num_problems: {num_problems}")
                print(f"groups: {groups}")
                
                # 调用 atmm 内核
                dispatch_sgmm(
                    out_tensor[slice_id],
                    data.inputs_tensor,
                    data.lora_weights[slice_id],
                    a_start, a_len, a_loc,
                    0, a_scaling,
                    output_counts, rank_counts, lora_ids, start_ids, tmp_d,
                    num_problems,32, 32, 32, 32, 32, 32)

    # Reference
    sgmv_shrink_for_nslices(
        nslices,
        data.inputs_tensor,
        data.lora_weights,
        ref_out_tensor,
        *sgmv_meta_args,
        scaling,
    )
    import pdb; pdb.set_trace()
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
        triton_ops.lora_expand(data.inputs_tensor,
                               data.lora_weights,
                               out_tensor,
                               *lora_meta.meta_args(token_nums=token_nums),
                               offset_start=0,
                               add_inputs=add_inputs)

    # Reference
    sgmv_expand_for_nslices(nslices,
                            hidden_size,
                            data.inputs_tensor,
                            data.lora_weights,
                            ref_out_tensor,
                            *sgmv_meta_args,
                            add_inputs=add_inputs)

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
    "batches": [1],
    "num_loras": [1],
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
@pytest.mark.parametrize("op_type", ["shrink"])
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
                                 seq_length=64,
                                 scaling=1)
    else:
        check_lora_expand_kernel(batches=batches,
                                 num_loras=num_loras,
                                 rank=rank,
                                 hidden_size=hidden_size,
                                 nslices=nslices,
                                 dtype=dtype,
                                 device=device,
                                 seq_length=1,
                                 add_inputs=True)


@pytest.mark.parametrize("batches", hs_test_params['batches'])
@pytest.mark.parametrize("num_loras", hs_test_params['num_loras'])
@pytest.mark.parametrize("rank", hs_test_params['max_ranks'])
@pytest.mark.parametrize("hidden_size", hs_test_params['hidden_sizes'])
@pytest.mark.parametrize("nslices", [1])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", SEED)
@pytest.mark.parametrize("op_type", ["shrink"])
def test_kernels_hidden_size(
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
    Tests SGMV and LoRA kernels.
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
                                 seq_length=64,
                                 scaling=1)
    else:
        check_lora_expand_kernel(batches=batches,
                                 num_loras=num_loras,
                                 rank=rank,
                                 hidden_size=hidden_size,
                                 nslices=nslices,
                                 dtype=dtype,
                                 device=device,
                                 seq_length=128,
                                 add_inputs=True)
