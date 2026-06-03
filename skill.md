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
