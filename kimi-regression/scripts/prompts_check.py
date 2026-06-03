#!/usr/bin/env python3
"""kimi-regression: prompt check — a clear table of the RAW output of every endpoint (base + LoRA),
with correct LoRA routing. Run ad-hoc against a live server, OR `run_kimi.sh` runs it per cell (after bench).

WHY IT EXISTS — the `.pt` acc test (`acc_capture.py`) is teacher-forced **prefill-only**, so it CANNOT see a
*decode*-accumulating corruption. The trtllm-LoRA `down-overlap` bug is exactly that (a coherent prefix that
collapses to `!!!!`). Proven: a down-overlap-ON server scores a CLEAN acc (prefill) yet generates `!!!!` here.
So this is the decode-path gate the acc can't be.

LoRA routing (per `entrypoints/openai/serving_base._parse_model_parameter`):
  - OpenAI `/v1/chat/completions` & `/v1/completions`: model="<base>:<adapter>"  (split on first ':')
  - `/generate`: lora_path="<adapter>"
  - model="<adapter>" ALONE does NOT route (no colon → adapter=None → output == base).
Prompts are chat-templated (raw greedy degenerates to repeated '!' on this instruct model).

Usage:
  python3 prompts_check.py [--port 30000] [--lora alpha] [--model <tokenizer/model path>] [--cell <label>]
Prints a markdown table; a trailing line flags a `!!!!` collapse (the down-overlap signature).
"""
import argparse
import json
import urllib.request


def post(base, path, body):
    try:
        req = urllib.request.Request(
            base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        return json.load(urllib.request.urlopen(req, timeout=120))
    except Exception as e:  # noqa: BLE001
        return {"_err": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="30000")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--lora", default="alpha", help='adapter name (empty "" = base-only check)')
    ap.add_argument("--model", default=None, help="tokenizer/model path (default: served model id)")
    ap.add_argument("--cell", default="")
    a = ap.parse_args()
    B = a.base_url or f"http://127.0.0.1:{a.port}"
    has_lora = bool(a.lora)

    ids = [m["id"] for m in json.load(urllib.request.urlopen(B + "/v1/models"))["data"]]
    model = a.model or next((m for m in ids if m != a.lora), ids[0])

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        def ct(p):
            return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] no chat template ({e}); using raw prompts (LoRA may EOS-first on greedy)")
        def ct(p):
            return p

    def chat(p, m):
        d = post(B, "/v1/chat/completions", {"model": m, "messages": [{"role": "user", "content": p}], "max_tokens": 48, "temperature": 0})
        return d["choices"][0]["message"]["content"] if "choices" in d else str(d)

    def comp(p, m):
        d = post(B, "/v1/completions", {"model": m, "prompt": ct(p), "max_tokens": 48, "temperature": 0})
        return d["choices"][0]["text"] if "choices" in d else str(d)

    def gen(p, lora):
        b = {"text": ct(p), "sampling_params": {"max_new_tokens": 48, "temperature": 0}}
        if lora:
            b["lora_path"] = a.lora
        d = post(B, "/generate", b)
        d = d[0] if isinstance(d, list) else d
        return d.get("text", str(d))

    def c(s):
        return repr(s)[:72].replace("|", "\\|")

    lm = f"{model}:{a.lora}"  # OpenAI colon syntax that actually routes the LoRA
    print(f"## Prompt check — cell={a.cell or '-'}  base={model}  lora={a.lora or '(none)'}")
    if has_lora:
        print(f"LoRA routing: /generate lora_path={a.lora!r}; OpenAI model={lm!r} (colon). model={a.lora!r} alone does NOT route.")
    print("| # | endpoint | base output | LoRA output |")
    print("|---|---|---|---|")
    garbage = 0
    PROMPTS = [
        "What is the capital of France?",
        "Name three primary colors.",
        "What is 2+2?",
        "Briefly, what is a neural network?",
        "Write one sentence about the ocean.",
        "List two programming languages.",
        "Who wrote Romeo and Juliet?",
        "What is the boiling point of water in Celsius?",
    ]
    for i, p in enumerate(PROMPTS, 1):
        for ep, fb, fl in [
            ("chat_completion", lambda: chat(p, model), lambda: chat(p, lm)),
            ("v1/completions", lambda: comp(p, model), lambda: comp(p, lm)),
            ("generate", lambda: gen(p, False), lambda: gen(p, True)),
        ]:
            b = fb()
            l = fl() if has_lora else "—"
            if "!!!!" in str(b) or (has_lora and "!!!!" in str(l)):
                garbage += 1
            print(f"| {i} | {ep} | {c(b)} | {c(l) if has_lora else '—'} |")
    print(f"\n{'⚠ GARBAGE (!!!!-collapse) — likely the down-overlap decode bug.' if garbage else 'no !!!!-collapse seen.'}")


if __name__ == "__main__":
    main()
