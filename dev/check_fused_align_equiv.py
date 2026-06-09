#!/usr/bin/env python3
"""opt1 correctness micro-check: fused merged-align vs the unfused fallback.

Runs IN-POD on GB300 (needs torch+CUDA+sglang). Validates the opt1 change in
``virtual_experts.py`` (admit shared_outer to the fused single-launch align path)
by comparing, for the SAME routing inputs, the output of:

  fused    : moe_lora_merged_align(...)                       # the kernel opt1 enables
  fallback : _fused_virtual_topk_ids(...) + native moe_align  # the old shared_outer path

The two must be routing-equivalent. We do NOT require sorted_token_ids to be
bitwise-equal (within-expert order is implementation-defined: atomic scatter vs
count-and-sort) — the meaningful invariant is, per expert bucket, the SAME set of
token slots, plus an identical token_lora_mask. num_tokens_post_padded is reported
but not asserted equal across paths (compact/bucket-count differences change only
the padding, not the routed work).

Covers BOTH:
  - shared_outer=True  (the NEW path opt1 turns on) — must be equivalent.
  - shared_outer=False (ep_local per-expert path)   — regression: must stay equivalent.

Usage (ad-hoc, like prompts_check.py):
  kubectl cp dev/check_fused_align_equiv.py <pod>:/tmp/check_fused_align_equiv.py
  kubectl exec <pod> -- python3 /tmp/check_fused_align_equiv.py
  # config knobs: --num-experts 128 --ep 4 --top-k 8 --bs 16 --block-size 16 --trials 50
"""
from __future__ import annotations

import argparse
import sys


def build_expert_token_map(sorted_token_ids, expert_ids, num_post_pad, block_size, numel):
    """expert_id -> sorted list of valid token slots, from an aligned routing buffer.

    Padding slots (value >= numel) and empty/sentinel blocks (expert < 0) are dropped,
    so the map captures exactly the routed (token, expert) work regardless of buffer
    length or within-block ordering."""
    sti = sorted_token_ids.tolist()
    eids = expert_ids.tolist()
    nblocks = int(num_post_pad) // block_size
    m: dict[int, list[int]] = {}
    for b in range(nblocks):
        if b >= len(eids):
            break
        e = eids[b]
        if e < 0:
            continue
        blk = sti[b * block_size : (b + 1) * block_size]
        for t in blk:
            if 0 <= t < numel:
                m.setdefault(e, []).append(t)
    return {e: sorted(v) for e, v in m.items()}


def run_case(torch, shared_outer, args, trial):
    from sglang.jit_kernel.trtllm_lora_temp.moe_lora_merged_align import (
        moe_lora_merged_align,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size as native_moe_align_block_size,
    )
    from sglang.srt.lora.trtllm_lora_temp.triton_ops.virtual_experts import (
        _fused_virtual_topk_ids,
    )

    dev = "cuda"
    M, top_k = args.bs, args.top_k
    num_experts = args.num_experts
    local_num_experts = num_experts // args.ep
    # exercise a non-zero EP rank so the owned-window mask is non-trivial
    rank = (trial % args.ep)
    local_expert_offset = rank * local_num_experts
    max_loras = 1
    bs_align = args.block_size

    g = torch.Generator(device=dev).manual_seed(1234 + trial)
    topk_ids = torch.randint(
        0, num_experts, (M, top_k), dtype=torch.int32, device=dev, generator=g
    )
    # token_lora_mapping in {-1, 0}: some rows have no active adapter (mask must drop them)
    tlm = torch.where(
        torch.rand(M, device=dev, generator=g) < 0.2,
        torch.full((M,), -1, dtype=torch.int32, device=dev),
        torch.zeros(M, dtype=torch.int32, device=dev),
    )
    numel = M * top_k

    # ---- fused (the path opt1 enables for shared_outer) ----
    f_sorted, f_eid, f_npp, f_mask, f_vne = moe_lora_merged_align(
        topk_ids,
        tlm,
        num_experts,
        shared_outer,
        max_loras,
        bs_align,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
        do_skip=True,
        compact=not shared_outer,
    )

    # ---- fallback (old shared_outer path: _fused_virtual_topk_ids + native align) ----
    vtopk, b_mask, b_vne = _fused_virtual_topk_ids(
        topk_ids,
        tlm,
        num_experts,
        shared_outer,
        max_loras,
        local_expert_offset=local_expert_offset,
        local_num_experts=local_num_experts,
    )
    b_sorted, b_eid, b_npp = native_moe_align_block_size(vtopk, bs_align, b_vne)

    torch.cuda.synchronize()

    problems = []
    if not torch.equal(f_mask.to(torch.bool), b_mask.to(torch.bool)):
        problems.append("token_lora_mask differs")

    fmap = build_expert_token_map(f_sorted, f_eid, f_npp, bs_align, numel)
    bmap = build_expert_token_map(b_sorted, b_eid, b_npp, bs_align, numel)
    if fmap != bmap:
        only_f = {e: fmap[e] for e in fmap if fmap.get(e) != bmap.get(e)}
        only_b = {e: bmap[e] for e in bmap if bmap.get(e) != fmap.get(e)}
        problems.append(
            f"expert->tokens differs: fused_only={only_f} fallback_only={only_b}"
        )

    return problems, int(f_npp), int(b_npp), f_vne, b_vne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-experts", type=int, default=128)  # Qwen3-30B-A3B
    ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)  # decode batch
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: torch import: {e}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available (run this IN-POD on GB300)")
        sys.exit(1)

    print(
        f"== fused-align equiv check  experts={args.num_experts} ep={args.ep} "
        f"top_k={args.top_k} bs={args.bs} block={args.block_size} trials={args.trials}"
    )
    n_fail = 0
    for shared_outer in (True, False):
        tag = "shared_outer" if shared_outer else "per_expert(ep)"
        case_fail = 0
        sample = None
        for t in range(args.trials):
            try:
                problems, fnpp, bnpp, fvne, bvne = run_case(torch, shared_outer, args, t)
            except Exception as e:  # noqa: BLE001
                print(f"  [{tag}] trial {t}: ERROR {type(e).__name__}: {e}")
                case_fail += 1
                n_fail += 1
                continue
            if problems:
                case_fail += 1
                n_fail += 1
                if sample is None:
                    sample = (t, problems, fnpp, bnpp, fvne, bvne)
        if case_fail == 0:
            print(f"  [{tag}] PASS  ({args.trials}/{args.trials} routing-equivalent)")
        else:
            t, problems, fnpp, bnpp, fvne, bvne = sample
            print(f"  [{tag}] FAIL  {case_fail}/{args.trials} mismatched")
            print(f"     first @trial {t}: {'; '.join(problems)}")
            print(f"     npp fused={fnpp} fallback={bnpp} | vne fused={fvne} fallback={bvne}")

    if n_fail:
        print(f"\nRESULT: FAIL ({n_fail} mismatches) — do NOT trust the fused path yet")
        sys.exit(1)
    print("\nRESULT: PASS — fused == fallback routing for shared_outer and per-expert")


if __name__ == "__main__":
    main()
