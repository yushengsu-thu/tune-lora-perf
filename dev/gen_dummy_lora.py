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
# Shared-outer MoE adapters (--experts-shared-outer-loras): the routed-expert LoRA is emitted in
# the 3D shared-outer layout instead of per-expert 2D. We AUTO-DETECT this from the server run's
# own flags — DL_LORA_EXTRA (the LORA_EXTRA string common.sh forwards) is grep'd for
# `--experts-shared-outer-loras`; if present, shared-outer weights are generated, otherwise the
# default per-expert weights are. DL_SHARED_OUTER (1/0) force-overrides the auto-detect. In
# shared-outer mode the per-expert checkpoint params ".mlp.experts.<i>.<proj>" collapse into one
# 3D stacked tensor "...mlp.experts.<proj>", and the OUTER projection is shared across experts
# (expert_dim=1) — exactly what SGLang's loader stacks/auto-detects (gate_up lora_A.shape[0]==1):
#   gate/up:  lora_A [1, r, H]       (SHARED)      lora_B [E, I, r]   (per-expert)
#   down:     lora_A [E, r, I]   (per-expert)      lora_B [1, H, r]   (SHARED)
# Attention q/k/v/o and the per-layer mlp.shared_expert stay plain 2D in BOTH modes.
#
# Inputs via env (set by the caller):
#   DL_OUT      output dir (e.g. /data/<model>-dummy-lora-r16)        [required]
#   DL_BASE     base model dir (headers only are read)                [required]
#   DL_RANK     LoRA rank r                                           [default 16]
#   DL_ALPHA    lora_alpha                                            [default = DL_RANK]
#   DL_TARGETS  comma list of HF module-name suffixes to target
#               [default q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
#   DL_STD      lora_B init std (nonzero => the adapter changes output) [default 0.02]
#   DL_LORA_EXTRA  the server's LoRA flag string; grep'd to auto-detect shared-outer [default ""]
#   DL_SHARED_OUTER  force shared-outer on(1)/off(0), overriding auto-detect          [default auto]
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

# Some checkpoints (e.g. the multimodal-wrapped Qwen3.5) name params "model.language_model.layers...",
# but PEFT adapters trained on them — and SGLang's LoRA loader mapping — use "model.layers..." (no
# "language_model" segment; see the real qwen35_35b_lora_alpha keys). Strip such wrapper segments so
# our dummy keys match real adapters exactly. No-op for checkpoints that don't carry the segment.
STRIP_SEGMENTS = ("language_model.",)


def normalize_base_name(n):
    for s in STRIP_SEGMENTS:
        n = n.replace(s, "")
    return n

# Shared-outer auto-detect: prefer the explicit DL_SHARED_OUTER override, else infer from whether
# the server run requests --experts-shared-outer-loras (forwarded as DL_LORA_EXTRA).
_so = os.environ.get("DL_SHARED_OUTER", "").strip().lower()
if _so in ("1", "true", "yes", "on"):
    SHARED_OUTER = True
elif _so in ("0", "false", "no", "off"):
    SHARED_OUTER = False
else:
    SHARED_OUTER = "--experts-shared-outer-loras" in os.environ.get("DL_LORA_EXTRA", "")

# A per-expert routed-expert checkpoint param: "...mlp.experts.<idx>.<proj>" (matched on the
# name with the trailing ".weight" stripped). Only these collapse to the 3D shared-outer layout.
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)$")


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
                shapes[normalize_base_name(n)] = shp
    return shapes


def main():
    if already_done():
        log(f"reuse existing adapter at {OUT}")
        return 0

    import torch
    from safetensors.torch import save_file

    log(f"base={BASE} rank={RANK} alpha={ALPHA} std={STD} shared_outer={SHARED_OUTER} "
        f"targets={TARGETS}")
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

    def init_A(shape):
        a = torch.empty(*shape, dtype=torch.float32)
        torch.nn.init.kaiming_uniform_(a, a=5 ** 0.5)
        return a.to(torch.bfloat16)

    def init_B(shape):
        return torch.normal(0.0, STD, size=tuple(shape), generator=gen).to(torch.bfloat16)

    if SHARED_OUTER:
        # Routed experts -> one 3D stacked tensor per (layer, proj), expert index collapsed out.
        # The outer projection is shared across experts (expert_dim=1); the loader stacks gate+up
        # and detects shared-outer from gate_up lora_A.shape[0]==1. Everything else stays 2D.
        groups = {}   # collapsed_stem -> {"experts": set(idx), "shape": (out, inp), "proj": str}
        n_plain = 0
        for n, (out, inp) in shapes.items():
            m = _EXPERT_RE.search(n[:-len(".weight")])
            if m:
                collapsed = re.sub(r"\.experts\.\d+\.", ".experts.", n[:-len(".weight")])
                g = groups.setdefault(
                    collapsed, {"experts": set(), "shape": (out, inp), "proj": m.group(2)})
                g["experts"].add(int(m.group(1)))
            else:
                stem = "base_model.model." + n[:-len(".weight")]
                tensors[stem + ".lora_A.weight"] = init_A((RANK, inp))
                tensors[stem + ".lora_B.weight"] = init_B((out, RANK))
                n_plain += 1
        for collapsed, g in groups.items():
            E, (out, inp), proj = len(g["experts"]), g["shape"], g["proj"]
            stem = "base_model.model." + collapsed
            if proj == "down_proj":          # down: per-expert A, shared outer B
                a_shape, b_shape = (E, RANK, inp), (1, out, RANK)
            else:                            # gate/up: shared outer A, per-expert B
                a_shape, b_shape = (1, RANK, inp), (E, out, RANK)
            tensors[stem + ".lora_A.weight"] = init_A(a_shape)
            tensors[stem + ".lora_B.weight"] = init_B(b_shape)
        n_exp = max((len(g["experts"]) for g in groups.values()), default=0)
        log(f"shared-outer mode: {len(groups)} routed-expert 3D modules (E={n_exp}), "
            f"{n_plain} plain 2D modules")
    else:
        for n, (out, inp) in shapes.items():
            stem = "base_model.model." + n[:-len(".weight")]
            tensors[stem + ".lora_A.weight"] = init_A((RANK, inp))
            tensors[stem + ".lora_B.weight"] = init_B((out, RANK))

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
