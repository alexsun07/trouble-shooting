# ROCm 上 tokenizer worker 在第一张卡上建 HIP context，KFD runlist 过订阅，scheduler kernel 周期性挂起 ~12 ms

一句话：KFD 的硬件调度器（HWS）每卡只能同时映射 24 个用户 compute queue，runlist 里**新进程排在前面**；
一旦别的进程比服务进程晚在同一张卡上建了队列、总数又超过 24，被挤出去的就是服务进程自己的队列，它的 kernel 每轮被挂起一个时间片。
sglang 里制造这些"晚来的队列"的，是多 tokenizer worker 在 warmup 时处理图片、在 `cuda:{base_gpu_id}` 上建的 HIP context。

- **实测环境**：MI355X ×8 / gfx950 / kernel 6.14.0 / ROCm 7.2.4，`lmsysorg/sglang:dev-minimax-m31-flash-mi35x` 的 python/
  换成支持 M3.1 的 sglang 开发分支，MiniMaxAI/MiniMax-M3.1-Flash，PP4 × TP1，`GPU_MAX_HW_QUEUES=16`。
- **负载**：75k 输入 / 1 输出、约 90% 前缀命中、并发 20/40，prompt 以文本形式发给 server。

## 现象

表面上看到的（不用 profiler）：

1. **吞吐明显低于预期，而且同一个配置每次重启都不一样**：多模态模型、`--tokenizer-worker-num` > 1 时，有的启动正常，
   有的启动整体慢一截，而且从启动到退出一直慢，不是只慢一阵（实测 MiniMax-M3.1-Flash PP4 prefill：tw=8 正常约 200k/卡，
   中招时 107k–175k/卡；tw=16 吞吐约为正常的 1/26）。等 warmup 结束、server 就绪后再压测也一样。
2. **`--tokenizer-worker-num` 越大越容易慢、越慢**；降到 1 反而"变快"（但 1 会被前端 tokenize 卡住，见 #2）。
3. **只有第一张可见卡忙**：`rocm-smi` 里第一张卡（PP0 / TP0）利用率长期 100%、功耗明显高于其他卡，PP 的后几张卡大部分时间
   利用率为 0（都在等它）；换一组 GPU 跑，慢的还是"第一张可见卡"，不是某张物理卡。
4. **第一张卡上多出几个"tokenizer"进程**：`rocm-smi --showpids` / KFD sysfs 里，若干个 `sglang::tokenizer_worker` 在第一张卡上
   各占约 682 MB 显存（其余只有 2 MB），但从不使用计算单元（`cu_occupancy` 一直是 0）。占 682 MB 的个数 ≥ 4 时就慢。
5. dmesg：`amdgpu 0000:xx:00.0: amdgpu: Runlist is getting oversubscribed due to too many queues. Expect reduced ROCm performance.`
   （PCI 地址就是第一张卡；只在"进入过订阅"那一刻打一次，所以行数和慢的程度无关，而且环形缓冲会滚动，要按时间戳数。）

trace 里看到的（torch profiler）：

6. 只有第一张卡的 rank 的 trace 里，任意类型的 kernel（GEMM、attention、MoE，甚至 40 us 的 elementwise）偶发被拉长，超出部分
   量子化在 11–14.5 ms，起点间隔也有规律（多出 1 个队列时每 66 ms 一次，多出 5 个时每 30–33 ms 一次）；没被打中的 kernel
   也慢几个百分点。计算流 100% 忙、没有 idle，所以看起来像"kernel 本身变慢"。其他 rank 干净，forward 时间只有它的一半左右。
   tw=4 实测一次 0 长尾、一次每 forward 4.6 次；tw=16 每次挂起约 206 ms，forward 从 0.24 s 变成 5.6–7.6 s。

## 快速判断

```bash
# 1. 第一张卡上每个进程的队列（type 0 = compute, 1 = SDMA）。tokenizer worker 有队列、总数 >24 就是这个坑
ssh <host> 'bash -s' < kfd_queues.sh
# 2. HWS runlist：是否过订阅（chain）、进程顺序（新的在前）。scheduler（17+2）排在最后、总数 >24 => 中
sudo cat /sys/kernel/debug/kfd/rls | python3 rls_decode.py -v
# 3. dmesg 有没有过订阅告警，PCI 地址是不是第一张卡
sudo dmesg | grep 'oversubscribed'
# 4. 每卡的用户 compute HQD 个数（MI300/MI355：每个 Inst 都是 Pipe 0-3 x Queue 2-7 = 24）
sudo grep -E '^ *Inst 0, +CP Pipe' /sys/kernel/debug/kfd/hqds | wc -l
```

要看是谁、在哪行代码建的 context：用 `kfdtrace/`（LD_PRELOAD + faulthandler，在 `open("/dev/kfd")` 和 KFD `CREATE_QUEUE`
时把所有线程的 Python 栈写到 `$KFDTRACE_DIR/trace.<pid>.txt`）：

```bash
gcc -O2 -shared -fPIC -o kfdtrace.so kfdtrace.c -ldl
export KFDTRACE_DIR=/tmp/kt LD_PRELOAD=$PWD/kfdtrace.so PYTHONPATH=$PWD:$PYTHONPATH   # 然后照常起服务
```

## 触发条件

**充要条件（实测）：同一张卡上 compute queue 总数 > 24，并且服务进程不是 runlist 里最靠前的那批**——也就是有进程
**比它晚**在这张卡上建了队列。

控制实验（tw=1 的服务 + 一个只建队列、之后不提交任何工作的辅助进程 `queue_hog.py`，只改两个变量）：

| 辅助进程建队列的时机 | 第一张卡 compute queue | 过订阅 | 长尾/forward |
|---|---:|---|---:|
| 服务启动前（排在 scheduler 后面） | 24 | 否 | 0 |
| 服务启动前 | 25 | 是 | 0 |
| 服务启动前 | 29 | 是 | 0 |
| 服务就绪后（排在 scheduler 前面） | 24 | 否 | 0 |
| 服务就绪后 | 25 | 是 | 3.0 / 211 ms，11.3 ms 每次、间隔 64 ms |

sglang 里的来源（实测，`kfdtrace` 栈）：每个 tokenizer worker 的 lifespan 起一个 warmup 线程，多模态模型
（`has_image_understanding`）时发一条 `/v1/chat/completions` + 32×32 PNG 到**同一个端口**，由内核分给任意一个 worker；
处理到图片的 worker 走 `base_processor._call_processor` → `_fast_image_processor_device()` 返回 `cuda:{base_gpu_id}` →
transformers `process_image` 里 `image.to(device)`，在第一张卡上建 context（2 compute + 1 SDMA，有时 3 compute + 0 SDMA）。
所以中招的 worker 数是随机的（tw=8 实测 4、6 个；tw=16 实测 10 个；tw=4 实测 3、4 个）。上线后真实的图片请求还会让更多 worker 建 context。

**warmup 是一次性的，但影响持续整个服务生命周期**：worker 处理完图片后，HIP context、它的 compute queue 和 PyTorch 缓存的
约 682 MB 显存都不会释放（HIP 没有归还队列的接口，进程退出才销毁），所以这些队列在 warmup 之后一直挂在 runlist 前面。
压测若只发文本，不会再新建 context，中招个数在 warmup 结束时就定了：总数 > 24 则这次启动从头到尾都慢，≤ 24 则整次启动都正常。
所以"等 warmup 结束、server 就绪后再压测"避不开这个问题。

**无关的维度（都实测排除过）**：
- 辅助队列是否空闲——完全空闲、rptr==wptr 不动的队列照样触发（只要排在前面）。
- 是否有 SDMA queue、是否用过 MemPool、进程个数——模仿 tokenizer 的 `tok_hog.py`（2+1 队列 ×3 个进程，结构和 4 个 tokenizer worker 一样）早建时 0 长尾。
- CPU 忙不忙 / KFD 进程驱逐——让模仿进程一直在 CPU 上 tokenize 也不触发；驱逐频率（dmesg 只读）比长尾低两个数量级。
  驱逐确实存在（KFD sysfs `evicted_ms` 各进程都在涨），但触发源是否为 NUMA balancing 导致的 userptr 失效**未经 root 级验证**
  （没做 ftrace、没改 `kernel.numa_balancing`），属推测；它不是本条的原因。
- 哪张物理卡、是不是别的用户——换到另一组 GPU，长尾跟着第一张可见卡走。
- sglang 分支——另一个分支的 warmup 和设备选择代码与实测分支相同，"另一分支 tw=4 没长尾"只是那次 k≤3。
- 其他 server 参数——`--max-prefill-tokens`（16384/32768）、`--reasoning-parser`/`--tool-call-parser`、`--fp8-gemm-backend`、
  `--mem-fraction-static`、`--kv-cache-dtype`：都不改变 tokenizer 进程是否建 context；mpt 32768 + tw=8（k=5）实测长尾照样在。
  step 大小也无关：所有配置的 step 都被 chunked-prefill-size 封顶在 32768。

## 根因

引入方（sglang 侧，判断为意外副作用）：上游 `34ef6c8135`（#5065，2025-04，tokenizer 进程里的 fast image processor 默认 `device="cuda"`）
与 `95876d75cb`（#9528，2025-11，VLM warmup 带一张图，多 tokenizer 模式下每个 worker 各发一次）叠加，让本不需要 GPU 的 tokenizer
进程在 scheduler 的卡上建 context。驱动侧行为如下：

- 内核（v6.14 `drivers/gpu/drm/amd/amdkfd/kfd_packet_manager.c`，`pm_calc_rlib_size` / `pm_create_runlist_ib`）：
  `active_cp_queue_count > get_cp_queues_num()` 时在 runlist 末尾加一个 chain 回自己的 `RUN_LIST` 包（过订阅），并打印上面的 dmesg。
  `get_cp_queues_num()` = amdgpu 留给 KFD 的 HQD：MI300/MI355 每个 XCC 4 pipe × queue 2–7 = 24（queue 0–1 给了 KCQ）。
- runlist 按 `dqm->queues` 链表顺序生成，**新注册的进程在前**（`rls_decode.py -v` 实测）。HWS 按顺序把队列映射到 HQD。
- 过订阅时放不下的队列要等时间片轮转；排在最后的服务进程的队列被换下，正在跑的 kernel 停一个时间片。
  实测停顿约 11–12 ms，KFD 在 MAP_PROCESS 里写的 `process_quantum = 10`；**时间片单位和 HWS 固件的轮转策略没有公开文档，这一段是推测**。
  tw=16 时（40 个队列、10 个晚来的进程）每次停约 206 ms ≈ 17 × 12.1 ms。
- `/sys/kernel/debug/kfd/mqds` 里 MQD 的 `cp_hqd_active` 佐证：晚来的空闲队列占着 HQD 时，scheduler 的队列多数时刻处于未映射；
  早建的空闲队列则相反，自己处于未映射，scheduler 全部映射。

## 修复

- **代码（推荐）**：让 tokenizer 进程不碰 GPU。补丁：ROCm 上
  `_fast_image_processor_device()` 返回 `"cpu"`（同一条 torchvision 路径），`SGLANG_ROCM_MM_PREPROCESS_ON_GPU=1` 可恢复旧行为。
  实测修复后 8 次图片请求过后 tokenizer worker 仍是 0 队列，第一张卡始终 17 个；MiniMax-M3.1-Flash PP4 prefill tw=4/8/16 都是约 200k/卡（aligned），
  原来 tw=8 是 107k。
- **没有补丁时**：`GPU_MAX_HW_QUEUES=8`（scheduler 只开 9 个队列）实测 0 长尾，但这个负载下 E2E 比修复后低 7–8%（PP0 forward 不变，
  推测是 PP 级间通信重叠变差）；tw ≤ 3 能让 warmup 最多建 3 个 context（17 + 6 = 23），但真实图片流量会让更多 worker 建 context，不保险。

**无效或有坑的直觉做法**：
- 只看"每卡队列数"然后去减队列：早建的进程再多队列也无害，判断要看 runlist 顺序（`rls_decode.py -v`）。
- 把 `--tokenizer-worker-num` 降到 1：长尾没了，但前端单线程 tokenize 成为瓶颈（实测 98k/卡，`#queue-req` 99% 为 0，见 #2）。
- 用 `HIP_VISIBLE_DEVICES=` 对 tokenizer 进程屏蔽 GPU：`import sglang` 时的平台探测和 aiter 导入都依赖 GPU 可见（`SGLANG_USE_AITER=1` 下实测 import 直接报 `aiter is required`），不可行。
- 只看 dmesg 行数估计严重程度：它只记录"进入过订阅"的次数。

## 影响范围

- 实测：MI355X、kernel 6.14、HWS（`sched_policy=0`、`mes=0`）、支持 M3.1 的 sglang 开发分支 + MiniMax-M3.1-Flash（多模态）+ PP4。
- 按机制推测也会中招：任何在 ROCm HWS 模式下"服务进程之后又有进程在同一张卡上建队列"的场景（例如在服务运行时于同一张卡上起
  profiler、监控、或者另一个用 GPU 的辅助进程），以及 TP 模式下的 TP0 卡；`GPU_MAX_HW_QUEUES` 越大，余量越小。
  MES 架构（gfx11/12）的调度方式不同，没有验证。

## 复现

1. 起一个 tw=1 的服务（第一张卡上 17 + 2 = 19 个 compute queue）。
2. 服务就绪后，在同一张卡上运行 `GPU_MAX_HW_QUEUES=16 python3 queue_hog.py 5`（得到 6 个 compute queue，总数 25）；另一次在服务**启动前**运行同样的命令。
3. 打 prefill 负载并采 torch profile，看第一张卡 rank 的 trace：前者每约 64 ms 一次约 11 ms 的长尾，后者没有。
   `rls_decode.py -v` 可以看到两种情况下 runlist 里 `6+0` 分别排在最前和最后。
