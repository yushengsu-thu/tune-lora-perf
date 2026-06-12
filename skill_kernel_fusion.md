# Kernel Fusion / Kernel 优化 Skill

对于 kernel 优化任务：

## 0. 立项前 sanity check（必做，5 分钟，能省几天）

动手前先用既有的 graph-OFF trace 跑 `dev/sanity_check_opt.py <trace> [--remove-us-per-layer X --layers N] [--kernel <name> --bytes <理论字节数>]`，回答三个问题：
- **该 phase 是不是 host-bound？**（PREFILL idle >40% ⇒ 砍 GPU 时间不会动 e2e，先去做 host 端：graphing / launch 批次化）
- **计划移除的 GPU 时间换算成 e2e 上限是多少？**（低于 ±2% 噪音底线 ⇒ 不立项，或改立 host 端项目）
- **目标 kernel 是 CONFIG-bound 还是 bandwidth-bound？**（实测/理论 HBM 时间 >3× ⇒ 是 launch 配置烂，修 grid（~30 行）即可，不需要融合手术）
实测教训（2026-06-12，bf16 LoRA opt 系列）：opt6/opt7 共 ~1,600 行 GPU 端优化全部过正确性与 kernel gate（−62% kernel time）但 e2e 归零——prefill 67% GPU-idle；而 permute 的 180µs 经分诊是 grid=[8,16,1]/11% 占用率的配置问题（理论流量 8.8µs），30 行 gather kernel 解决。

## 1. Make a testbed: a script that can bench the kernels w/ correct shapes

### i. 基于 print 的理解问题：在相关 kernel 处，做充分的 print，理解所有输入输出的 shape、dtype 等信息

- remarks
    - sgl launch 时，可能需要 `--disable-cuda-graph`
    - print 至少包含：（1）所有输入、所有输出（2）shape、dtype、stride、device（3）除非无法使用 请用 `dumper.py :: get_tensor_info` 来获得充分信息
    - 如果你同时在处理多个 kernel，则这一步一般只需要跑一次 sglang engine
    - 注意你的 adhoc 代码，比如这些 print，都必须 commit，之后再 revert，留痕；这种 commit 不需要 format code
    - 一般引擎启动有 warmup，于是你的 print 已经会输出一堆东西了
    - 也可以用 bench one batch server 之类的工具，但注意要让 output len 比如是 5 这种小数字，避免大量 print
    - 注意区分 prefill 还是 decode，这种 warmup 一般开头 forward pass 是 prefill，后面几个是 decode，我们可能主要希望 decode
    - 注意不要任何去重，就朴素地 print 即可，我们不在乎日志大一点，但很在乎东西丢了
    - 一般你要 print 单行而不是多行，避免互相交错；（不过有的时候还是会交错）
- 产物
    - （1）有 unique name 的日志文件，拉到本地 artifacts 子文件夹，用于本次使用和永久存档（2）一个独立 md 报告，精确描述 launch command、使用的 public commit id、原始 log、分析后的 shape
    - 你理解成，我需要把这 1 个 md 交给别人，让人只用这个 md（和公开的 commit 之类）独立验证我们说的对不对，所以你要给完整的证据链

### ii. 构造 testbed (perf bench + correctness test 脚本)

- 参考 https://github.com/jybsuper/sglang/pull/12 :: `benchmark/kernels/lora_moe_expand/bench_expand_add_down.py` 这个 bench 脚本，对你的 kernel 写一个 self-contained 的脚本
- 其中至少包括
    - i. benchmarking (speed test)
    - ii. correctness test：如果方便，需要写一个 ref code，然后去验证我们的 kernel 和 ref code 保持一致；对于很多 kernel 要 assert bitwise equal、对于实在没办法的 也要 assert 数值误差很小。

## 2. Verify the testbed

- i. ensure the testbed's reported speed is the same as what we see in profiles
    - 请你询问人类，看到 profile 的速度是多少，你不需要自己做 profiling；如果人类没有回答/你不能问人类，请你直接跳过这一条，但标注说没有验证
- ii. ensure the testbed's shape is the same as what we printed in e2e cases
- iii. 用 subagent 审核写的 testbed 代码，一个 testbed 用一个 opus + 一个 codex 独立审核，审查正确性

## 3. Optimize the kernel, until it is fast

- i. 有了这些信息（e.g. shape），先分析一个理论速度上界、roofline 之类，写成一个独立 md 报告
- ii. 和人类讨论你的优化思路，有一个大致方向后，再写一个独立 md 报告；这个方向以后还可以改（甚至完全抛弃自主换方向），只是第一步先暂时定一个
- 附：开始用 gpu 前，和人类确认使用方式。
    - 例如，人类可能会说：
        - 远程 pod 正有多个 agent 共享使用
        - 如果不希望被干扰，对于跑 jit kernel 测试，可以在远程开一个独立临时目录，git clone 之类更新你的代码，然后 `PYTHONPATH=/path-to-your-sglang/python python your_script.py`，而不需要用 uv 等虚拟环境。
        - 同时对于本地，你可以开一个 worktree，里头是个临时 branch，然后记得 atomic commit + push
- iii. 构造正确性测试护栏
    - a. 如果你在写新 kernel（比如 fusion），那老 kernel 就是你天然的测试护栏，你只要在 bench 脚本中加一节，新 kernel vs 老 kernel 对比，即可
        - 对于很多 kernel 要 assert bitwise equal、对于实在没办法的 也要 assert 数值误差很小
    - b. 如果你在魔改老 kernel，则需要保证存在 ref code，且老 kernel 与 ref code 之间已经有充分的对拍正确性测试了
- iv. 开一个 `environ.py` 的 env var，打开/关闭则走新/旧路径，便于方便地二分测试
    - 简单起见，即使你在改老 kernel，除非改动很小，也可以直接复制粘贴出来作为一个新 kernel 来做，不用担心 code duplication
- v. 尝试优化，可以反复多次尝试、尝试新的优化思路之类，直到速度快
- vi. 交付前检查
    - 确认最新代码跑通了 correctness test + 跑出了符合期待的 bench 结果
    - 开 opus、codex subagents 独立审查你的改动的正确性
