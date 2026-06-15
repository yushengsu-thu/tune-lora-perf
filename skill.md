1. 你看目前的任務是什麼？在/Users/yushengsu/Downloads/river的路徑下以任務名稱創建一個路徑，然後這路徑裡生成一個journal.md，這個journal.md記錄你要做的事、你做的每一步，以及每次跑acc跟benchmarking的結果。
2. 然後我目前的repo是 /Users/yushengsu/Downloads/river/sglang, branch 是 lora-opti（本機分支，tracking 上游 https://github.com/jybsuper/sglang/tree/lora-opti），你每次都要檢查一下是否 lora-opti 都有跟原本的branch，最新的code: https://github.com/jybsuper/sglang/tree/lora-opti sync，如果沒有sync的話你都要sync (git fetch jybsuper && git switch lora-opti && git merge --ff-only jybsuper/lora-lora)，然後你基於這個branch創建一個新的branch，新的branch的名字依照這任務命名，然後要注意的是，可能多個agent會共用同一個repo，所以你要自己建立worktree。
3. 你的任務應該有多個steps，每做一個step，你就把這個新branch以及修改的code上傳到你launch的nodes
4. id=随便起个名字就行, 我用来防止k8s resources 有conflict用的, 我的id用yushengsu-${date}-${time}
5. 把一組你嘗試的都寫到pr的decription裡面，還有把journal.md也上傳上去，再用這些做法跑 regression跟benchmark。
6. 然后你啟動node跑 /Users/yushengsu/Downloads/river/sglang-base-variant-regression.md 檢查一下，如果修改失敗就debug到修復為止，這是測試Qwen模型的skill，如果要測試kimi的，需要用kimi-regression，如果通過就到下一個。
7. 然后你啟動node跑 /Users/yushengsu/Downloads/river/sglang-lora-base-perf-benchmark.md 檢查一下，如果失敗就debug到修復為止，如果通過就到下一個，在這邊 Qwen跟 Kimi的模型都要測試。
8. 做完benchmark跟精度測試就把nodes釋放掉。 如果上面都通過的話，就幫我push一個commit，執行下一步，然後到2。如果任務完成（每一步都做完的話）就幫我push這個pr

---

## ⚠️ 規則（DECODE-THPT-RULE）：跑 benchmark 不能只看 e2e 結果
禁止只看 bench 的 e2e 匯總數字（throughput / latency 那行）就下結論。**必須同時去看 server log 裡打印出來的 decode throughput（gen/decode token/s）**，確認它跟 e2e 結果一致，並把這個 decode thpt 數字一起記錄到 journal.md / PR description 裡。
**（已自動化，2026-06-12）**：`dev/bench_profile_matrix.sh` 現在每個 bench cell 自動切 server-log 切片
（`bs<N>.serverlog`）並跑 `dev/serverlog_sanity.py`（差 >5% 打 SUSPECT 警告）；profile 拉回後自動跑
`dev/profile_metrics.py` 作獨立見證。看到 `[SANITY] ... SUSPECT` 必須重跑該 cell 再判讀方向。

---

## ⚠️ 規則（OPT-PAYOFF-RULE）：任何 kernel/perf 優化立項前，先跑收益存在性檢查
動手寫優化之前，先用既有的 graph-OFF trace 跑 `dev/sanity_check_opt.py`：
1. **host-bound 判定**（PREFILL 行 idle >40% ⇒ GPU 端優化不會動 e2e，只有 host 端工作會）；
2. **e2e 天花板**（`--remove-us-per-layer X --layers N`，上限低於 ±2% 噪音 ⇒ 不立項）；
3. **kernel 分診**（`--kernel <name> --bytes <理論流量>`，實測/理論 >3× ⇒ CONFIG-bound，
   修 launch config（~30 行）而不是融合手術）。
教訓出處：opt6（−34% act kernel）/opt7（−62% MoE pipeline）兩個 GPU 端優化 e2e 全部歸零
（prefill 67% GPU-idle、host-bound）；而 permute「180µs」其實是 grid 配錯（理論流量僅 8µs），
30 行 gather kernel 取代了原計畫的多日 cpasync mainloop 手術。
---

## ⚠️ 規則（ACC-REFERENCE-RULE）：acc 的 reference data 要跟 adapter 種類匹配
跑 acc / KL-vs-reference 比較時，**reference 資料的選擇取決於 adapter 有沒有
`experts_shared_outer_loras` tag**（https://github.com/sgl-project/sglang/pull/21466 的語義，
看 adapter_config.json）：
- **一般 adapter**（如 `jybsuper/qwen35_35b_lora_alpha`）→ 用 adapter repo **內附**的
  `compare_sample_train_data.pt`（dev/4_run_acc.sh 的預設）。
- **有 `experts_shared_outer_loras` tag 的 adapter** → 才用
  `hf.co/datasets/yushengsu/datasets` 裡的 reference（`ACC_HF_FILE=<file>`）。

用錯 reference 時 KL-vs-reference 表**沒有意義**（實測 2026-06-06：KL≈0.42–0.52 vs floor
0.0006，連 no-lora cell 都超大 — 這不是 LoRA 路徑壞掉，是 reference 不可比）；
lora-vs-no-lora 那張表不受影響。`dev/4_run_acc.sh` 會自動偵測 tag 並擋下錯配
（`ACC_FORCE=1` 可強制覆寫）。

---

## ⚠️ 規則（STUCK-CHECK-RULE）：看起來卡住 ≠ 真的卡死，先驗證再動手
當一個 launch/run/autotune **看起來卡住**（log 不動、某步 elapsed 一直往上爬、進度條停在同一步），**不要**直接假設它死了就 kill/重啟。要**時不時用 `top`/`htop`/`nvidia-smi` 之類去看**它到底在不在跑：
- `top`/`htop`：CPU 在忙 → 通常是 JIT/編譯/autotune 在跑（例如 cold `fp4_gemm` autotune 是 **CPU-bound**，這時 **GPU util 會是 0%**，但它沒卡）。
- `nvidia-smi`：看 util / power / memory 有沒有變化。
- 看 **log 的位元組數或步數**在 ~60–120s 內有沒有往前（`wc -c`、進度條的 step 數）。
- 只有在 **CPU≈0 且 GPU util≈0 且 log/step 連續兩次檢查都沒推進**時，才判定為真 hang，再走 kill_all + 乾淨重啟。
（血淚教訓：曾把 cold autotune 的 0% GPU 誤判成 hang。）

---

## 🔥 規則（AUTOTUNE-CACHE-SEED）：autotune 結果存哪 + 怎麼預熱所有 node
fp4_gemm 的 cold autotune 要 ~20 分鐘，結果存在每個 pod 的：
`/root/.cache/sglang/flashinfer/autotune/<ver>/sm100/<hash>/rank_tp{0..7}_pp0_dp0.json`
（完整 ≈ 206 entry；**TP8 與 EP8 是不同 hash**；head pod 存 tp0–3、worker 存 tp4–7）。配套 JIT 快取在 `/root/.cache/{flashinfer,torch,tvm-ffi}`。

**持久化**：pod spec 用 hostPath 把 `/root/.cache` → node 本地 `/var/lib/sglang-cache`（`DirectoryOrCreate`），autotune cache 跨 pod 重建仍在，~20 分冷 tune 每個 node 只付一次。⚠️ **只對新建的 pod 生效**；既有 pod 仍用各自 ephemeral `/root/.cache`。

**一鍵預熱所有 GB200 node**（從一個已跑完的 cache）：pull 兩個 pod 的 cache 合併 → 臨時 DaemonSet 鋪到每個 node：
```bash
kubectl exec mnnvl-kimi-$ID-0 -- bash -lc 'cd /root/.cache && tar czf - sglang flashinfer torch tvm-ffi' > pod0.tgz
kubectl exec mnnvl-kimi-$ID-1 -- bash -lc 'cd /root/.cache && tar czf - sglang' > pod1.tgz
mkdir stage && tar xzf pod0.tgz -C stage && tar xzf pod1.tgz -C stage    # union → 8 ranks per hash
tar czf cache-seed.tgz -C stage sglang flashinfer torch tvm-ffi          # 不含 huggingface(模型)/pip
kubectl apply -f - <<'YAML'
apiVersion: apps/v1
kind: DaemonSet
metadata: {name: sglang-cache-seed, labels: {app: sglang-cache-seed}}
spec:
  selector: {matchLabels: {app: sglang-cache-seed}}
  template:
    metadata: {labels: {app: sglang-cache-seed}}
    spec:
      tolerations: [{operator: Exists}]      # 落到全部 node（arm64+gpu taint，含 cordoned）
      terminationGracePeriodSeconds: 0
      containers: [{name: seed, image: busybox:stable, command: ["sh","-c","sleep 3600"],
        volumeMounts: [{name: c, mountPath: /seed}]}]
      volumes: [{name: c, hostPath: {path: /var/lib/sglang-cache, type: DirectoryOrCreate}}]
YAML
kubectl get pods -l app=sglang-cache-seed -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
  while read p; do kubectl exec -i "$p" -- tar xzf - -C /seed < cache-seed.tgz; done
kubectl delete ds sglang-cache-seed --grace-period=0
```
驗證：每個 node `/var/lib/sglang-cache/.../sm100/<hash>/` 應有 8 個 `rank_tp*.json`，**TP8+EP8 兩組都要**。Seed 很小(~0.5MB) —— autotune 結果就是那些 JSON，順手帶 JIT 目錄省重編。
**zsh 坑**：未加引號的 `$VAR` 不會 word-split → 用 `while read` 迭代 pod，不要 `for p in $PODS`。

---

## ⚠️ 規則（PR-DESC-SYNC-RULE）：改了 code 就要同步 PR #4 的敘述與圖
只要你改動的 code 會落到 **PR #4**（`qwen3-30b-a3b-2507-bf16` → `trtllm-lora-bf16`，
<https://github.com/yushengsu-thu/sglang/pull/4>），**push 後必須對應修正 PR #4 的 description**
（body 錨點 `#issue-4607084295`），讓敘述與圖跟實際 code 一致：
- **敘述**：`## Summary` / `## Files changed (what & why)` / `## Verification`（GB300 launch/acc/bench 數字）/
  `## Notes` —— 行為、檔案清單、驗證數字變了就跟著改。
- **圖**：`## Flow` 裡的 ASCII pipeline 圖（`1) Normal LoRA` 與 `2) Shared-outer (TML) LoRA`，各自的
  `(A) Single-stream` / `(B) Two-stream`）—— 只要 kernel／管線結構變了（融合、stream 安排、新增或移除
  kernel、permute/activation/shrink/expand 的順序），這些流程圖要重畫到吻合。
- **做法**：`gh pr view 4 --json body --jq .body > /tmp/pr4.md` → 編輯 → `gh pr edit 4 --body-file /tmp/pr4.md`。
- **判斷範圍**：改變**預設管線**（default-on 行為）才要動 Flow 圖；default-OFF／flag-gated 的東西
  （如 opt6/opt7、opt8 piecewise）在 `## Notes` 註明即可，不必重畫主流程圖。
教訓：PR description 是這條 bf16 LoRA path 的單一事實來源；code 與敘述／圖一旦不同步，review 跟交接都被誤導。

---

## ⚠️ 規則（PROFILE-RELEASE-RULE）：profile trace 一律進 GitHub Release，且必須上傳
torch profiler 的 `*.trace.json.gz`（graph-on/off + gpu-only + perfetto，每份 ~數十 MB、一輪 8-rank
可達 ~1 GB）**不進 git repo**（會永久膨脹每個 clone），而是上傳成該 run 的 **GitHub Release 資產**
（tag `<model>-<DATE-TIME>`，repo `yushengsu-thu/lora_perf_lora_profile`，2 GiB/檔上限）。
- repo 的 `runs/<model>/<DATE-TIME>/` 只放**小檔**：bench summary/jsonl/serverlog、`gpu_busy_witness`
  輸出、bench.log、README（README 內含 release 連結 + `gh release download <tag> -R <repo>` 指令）。
- `6_upload_results.sh` 已自動做這個切分。**profile 上傳是強制的**：trace 一定要進 release，
  網路截斷就重試(pod pull 本來就會retry)，**不可略過 profile**。
- 資產命名 `cell__graph_mode__file`（路徑斜線→`__`，扁平且唯一）。
- **profile 目錄結構要保留成 README stub**：每個原本放 trace 的目錄（`<cell>/graph_<mode>/`、
  `gpuonly`）在 repo 裡仍然存在,但裡面放一個 `README.md` 指向 release(列出該目錄原本的 trace
  對應到哪些 release 資產 + `gh release download` 指令)。**目錄不可因為檔案搬走就消失**——這樣在
  `runs/<model>/<DATE-TIME>/current_base_lora/lora/graph_on/` 瀏覽時仍找得到 trace 的去向。
  `6_upload_results.sh` 已自動寫這些 stub。
教訓：~92 MB 的 graph-on trace 直接 commit 進 repo,history 會被永久撐大;release 資產存在 git 之外,
clone 不受影響——這是「省空間」的正解,不是 git-LFS(配額/頻寬更糟)。

---

## ⚠️ 規則（REPO-HYGIENE-RULE）：一次性 script 不留在 repo
為某次 debug / 某個 opt 的 A-B / 某次 setting 跑而寫的**一次性 script** 不留在 repo —— 結論進
`journal.md` / PR description，script 本身用完即刪（git history 仍可還原）。
- **不留**：`diag_*`（卡住/recompile/stream 診斷）、`probe_*`（cubin/piecewise 探測）、`exp_*`
  （numa/capture 實驗）、opt 專屬 A-B（`bench_opt<N>_ab` / `profile_opt<N>_*` / `profile_matrix` /
  `bench_3way` / `reprofile_*` / `profile_base_lora` / `profile_decode_bystage`）、單次 profile dump
  的 `.md`（`*_kernel_shapes_*` / `*_profile_graphon_*`）。
- **canonical harness（留）**：`dev/` 的 `1_`–`8_` 編號流程 + `run_all.sh` + `common.sh` /
  `jit_store.sh` / `jit_fp.cmd`；**唯一**的 per-opt 量測入口 `dev/bench_profile_matrix.sh`；sanity 三件套
  `sanity_check_opt.py` / `serverlog_sanity.py` / `profile_metrics.py`（+ `check_fused_align_equiv.py`）；
  `gpu_busy_witness.py`、`convert_to_perfetto_compatible.py`（5_run_profile 依賴）；`acc_*_capture.py`、
  `gen_dummy_lora.*`。
- **opt 專屬 parity test**（如 `test_bf16_*`）是可重跑的驗證、且被對應 PR 引用 → **保留**，但要綁定該 PR/journal，別變孤兒。
教訓：dev/ 曾累積 ~20 個一次性 diag/probe/opt-A-B script 與 canonical 流程混在一起，交接時分不清哪個是真入口。
