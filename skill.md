1. 你看目前的任務是什麼？在/Users/yushengsu/Downloads/river的路徑下以任務名稱創建一個路徑，然後這路徑裡生成一個journal.md，這個journal.md記錄你要做的事、你做的每一步，以及每次跑acc跟benchmarking的結果。
2. 然後我目前的repo是 /Users/yushengsu/Downloads/river/sglang, branch 是 lora-opti-nvfp4（本機分支，tracking 上游 https://github.com/jybsuper/sglang/tree/nvfp4-lora），你每次都要檢查一下是否 lora-opti-nvfp4 都有跟原本的branch，最新的code: https://github.com/jybsuper/sglang/tree/nvfp4-lora sync，如果沒有sync的話你都要sync (git fetch jybsuper && git switch lora-opti-nvfp4 && git merge --ff-only jybsuper/nvfp4-lora)，然後你基於這個branch創建一個新的branch，新的branch的名字依照這任務命名，然後要注意的是，可能多個agent會共用同一個repo，所以你要自己建立worktree。
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
