#!/usr/bin/env python3
# dev/gen_dummy_lora.py — generate a random-init ("dummy") PEFT LoRA adapter for perf testing.
#
# RUNS ON THE POD (piped to the pod's python by gen_dummy_lora.sh / common.sh's ensure_dummy_lora),
# where the base checkpoint already lives on /data.
#
# Approach — MIRROR THE BASE CHECKPOINT (no model load, no PEFT, no FP8 kernels):
#   We read only the safetensors *headers* of the base model (shapes, not data) and, for every base
#   weight whose module suffix is a LoRA target, emit a matching lora_A [r, in] + lora_B [out, r].
#   Names mirror the base param names exactly (PEFT prefix `base_model.model.`), so SGLang's loader
#   maps them by layer-id + substring just like a real trained adapter. This covers, automatically:
#     - attention q/k/v/o on the full-attention layers,
#     - the per-layer mlp.shared_expert gate/up/down,
#     - ALL routed experts (mlp.experts.<i>.{gate,up,down}_proj) — the --lora-use-virtual-experts
#       MoE path, which is the whole point of the perf work (~2.4G adapter at rank 16, like the real one).
#
# Why not meta+PEFT: AutoModelForCausalLM.from_config gives a TEXT-only module tree
# (model.layers..., no `language_model`) that (a) diverges from the served checkpoint's names and
# (b) packs the routed experts into a fused module PEFT can't target — so it silently misses the
# 256-expert MoE path. Mirroring the checkpoint is exact and complete.
#
# Excluded on purpose: `mtp.*` (MTP/speculative layers — get_layer_id would alias them onto real
# layer 0 and corrupt it; not served unless spec-decoding is on) and `*.visual.*` (vision tower —
# SGLang's text LoRA doesn't wrap it; its suffixes don't match anyway).
#
# Inputs via env (set by the caller):
#   DL_OUT      output dir (e.g. /data/<model>-dummy-lora-r16)        [required]
#   DL_BASE     base model dir (headers only are read)                [required]
#   DL_RANK     LoRA rank r                                           [default 16]
#   DL_ALPHA    lora_alpha                                            [default = DL_RANK]
#   DL_TARGETS  comma list of HF module-name suffixes to target
#               [default q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
#   DL_STD      lora_B init std (nonzero => the adapter changes output) [default 0.02]
#
# Idempotent: if DL_OUT already holds a complete adapter it exits 0 without rewriting.

import glob
import json
import os
import re
import sys

OUT = os.environ["DL_OUT"]
BASE = os.environ["DL_BASE"]
RANK = int(os.environ.get("DL_RANK", "16"))
ALPHA = int(os.environ.get("DL_ALPHA", str(RANK)))
STD = float(os.environ.get("DL_STD", "0.02"))
TARGETS = [t for t in os.environ.get(
    "DL_TARGETS", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",") if t]
TARGET_SET = set(TARGETS)
EXCLUDE = ("mtp.", "model.visual.", "visual.")


def log(*a):
    print("[dummy-lora]", *a, flush=True)


def already_done():
    cfg = os.path.join(OUT, "adapter_config.json")
    wts = os.path.join(OUT, "adapter_model.safetensors")
    return os.path.isfile(cfg) and os.path.isfile(wts) and os.path.getsize(wts) > 0


def module_suffix(weight_name):
    # strip trailing ".weight", return the last dotted component (the module name)
    base = weight_name[:-len(".weight")] if weight_name.endswith(".weight") else weight_name
    return base.rsplit(".", 1)[-1]


def base_weight_shapes():
    """Return {weight_name: [out, in]} for every base linear weight we will mirror,
    reading ONLY safetensors headers (no tensor data is loaded)."""
    from safetensors import safe_open

    idx = os.path.join(BASE, "model.safetensors.index.json")
    if os.path.isfile(idx):
        weight_map = json.load(open(idx))["weight_map"]
        shards = {}
        for n, f in weight_map.items():
            shards.setdefault(f, []).append(n)
    else:
        files = glob.glob(os.path.join(BASE, "*.safetensors"))
        if not files:
            raise FileNotFoundError(f"no safetensors found under {BASE}")
        shards = {os.path.basename(f): None for f in files}  # None => all keys in that file

    shapes = {}
    for fname, keys in shards.items():
        with safe_open(os.path.join(BASE, fname), framework="pt") as f:
            if keys is None:
                keys = list(f.keys())
            for n in keys:
                if not n.endswith(".weight"):
                    continue
                if any(n.startswith(p) or p in n for p in EXCLUDE):
                    continue
                if module_suffix(n) not in TARGET_SET:
                    continue
                shp = list(f.get_slice(n).get_shape())
                if len(shp) != 2:           # only plain 2D linears get a standard LoRA
                    continue
                shapes[n] = shp
    return shapes


def main():
    if already_done():
        log(f"reuse existing adapter at {OUT}")
        return 0

    import torch
    from safetensors.torch import save_file

    log(f"base={BASE} rank={RANK} alpha={ALPHA} std={STD} targets={TARGETS}")
    shapes = base_weight_shapes()
    if not shapes:
        log(f"ERROR: no base weights matched targets {TARGETS} under {BASE}")
        return 3

    # Report coverage so the operator can sanity-check (experts vs attention vs shared_expert).
    def bucket(n):
        if ".experts." in n:
            return "routed_experts"
        if ".shared_expert." in n:
            return "shared_expert"
        if ".self_attn." in n:
            return "attention"
        return "other"
    cov = {}
    for n in shapes:
        cov[bucket(n)] = cov.get(bucket(n), 0) + 1
    log("base modules to adapt:", sum(cov.values()), cov)

    tensors = {}
    gen = torch.Generator().manual_seed(0)
    for n, (out, inp) in shapes.items():
        stem = "base_model.model." + n[:-len(".weight")]
        a = torch.empty(RANK, inp, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        b = torch.normal(0.0, STD, size=(out, RANK), generator=gen)
        tensors[stem + ".lora_A.weight"] = a.to(torch.bfloat16)
        tensors[stem + ".lora_B.weight"] = b.to(torch.bfloat16)

    os.makedirs(OUT, exist_ok=True)
    save_file(tensors, os.path.join(OUT, "adapter_model.safetensors"))

    present = sorted({module_suffix(n) for n in shapes})
    cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": RANK,
        "lora_alpha": ALPHA,
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "target_modules": present,
        "modules_to_save": None,
        "base_model_name_or_path": BASE,
    }
    with open(os.path.join(OUT, "adapter_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    sz = os.path.getsize(os.path.join(OUT, "adapter_model.safetensors")) / 1e9
    log(f"wrote {OUT}: {len(tensors)} tensors ({len(shapes)} modules) "
        f"{sz:.2f} GB bf16  target_modules={present}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
