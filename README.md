# 防踩坑指南

注意：如果要添加新的记录，请确保已经弄清楚触发场景并定位清楚根因，记录的应该尽可能提供给更general的使用范围而不局限于某一个机器、环境。如果是特有环境才会触发，也应该加以说明。

## 目录

| # | 问题 | 详细说明 |
|---|---|---|
| 1 | ROCm 上 torch profiler 抓不到 GPU kernel | [1_rocm_torch_profiler_no_gpu_events/](1_rocm_torch_profiler_no_gpu_events/) |
| 2 | vLLM 前端 tokenize 单线程，把引擎饿住 | [2_vllm_renderer_num_workers_frontend_bottleneck/](2_vllm_renderer_num_workers_frontend_bottleneck/) |
| 3 | sglang MXFP8 scale 改名静默丢掉 NVFP4 expert block scale，输出乱码 | [3_sglang_mxfp8_scale_rename_drops_nvfp4_expert_scales/](3_sglang_mxfp8_scale_rename_drops_nvfp4_expert_scales/) |
| 4 | sglang PD（mooncake）长时间压测后临时端口耗尽，prefill 报 `Cannot assign requested address` | [4_mooncake_tcp_aux_ephemeral_port_exhaustion/](4_mooncake_tcp_aux_ephemeral_port_exhaustion/) |
| 5 | ROCm 上 sglang 的 tokenizer worker 在第一张卡上建 HIP context，KFD runlist 过订阅，scheduler kernel 周期性挂起 ~12 ms | [5_rocm_kfd_runlist_newer_process_starves_scheduler/](5_rocm_kfd_runlist_newer_process_starves_scheduler/) |

---

### 1. ROCm 上 torch profiler 抓不到 GPU kernel

- **现象描述**：这个 bug 几乎从不以"profiler 坏了"的形式出现，而是伪装成别的问题——**两类典型案例**：① **kernel tuning**（aiter `gemm_moe_tune.py`）所有候选 `us=0`、刷屏 `no valida data after post process!`、最终 `tune 0 shapes`，看着像 kernel 不支持或数据构造有问题，实际 kernel 完全正常（`cuda.Event` 实测 860–950 µs）、空的是采集不到 event 的 profiler；② **采集 profile trace**（sglang `/start_profile`）trace 文件正常生成且无任何报错，但里面只有 `python_function` / `cpu_op`、`kernel` 类别为 0。
- **触发场景**：进程启动时环境里已带着 `ROCPROFILER_REGISTER_LIBRARY`——最常见的是父进程调用过 `torch.cuda.is_available()`（不需要跑任何 kernel）之后再起子进程，因此凡是"父进程查一下 GPU → spawn worker 干活"的工具都可能中招（实测：sglang 的 scheduler、aiter tuner 的 worker），而设置该变量的进程**自身不受影响**。
- **根因**：rocprofiler 在第一次触碰 HIP 时用 C 层 `setenv()` 注入该变量，后代进程继承后其注册逻辑判定"已注册"而走短路分支，kineto 收不到 dispatch 回调；该变量对 Python 的 `os.environ` 不可见，所以排查时对比 `os.environ` 会误报"零变化"。
- **解决方案**：用 `sitecustomize.py` 在子进程解释器启动时（早于 HIP init）`os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")` 并挂到 `PYTHONPATH`（可直接取用 [1_rocm_torch_profiler_no_gpu_events/sitecustomize.py](1_rocm_torch_profiler_no_gpu_events/sitecustomize.py)），注意**在 shell 里 `unset` 无效**（那时变量还不存在）——这一条对采集 trace 和 aiter tuning 两个场景都实测有效；若只需计时而不需要 trace，也可以改用 `torch.cuda.Event` 绕开 profiler，两条路实测选出同一个 kernel，详见 [详细说明](1_rocm_torch_profiler_no_gpu_events/)。

### 2. vLLM 前端 tokenize 单线程，把引擎饿住

- **现象描述**：长 prompt 的 serving bench 里三个信号同时出现——① **吞吐上不去但 GPU 很忙**（trace 里 GPU busy 95~98%，kernel 本身不慢）；② 服务日志的 `Engine ... Running: N reqs, Waiting: M reqs` 里 **`Running` 明显低于并发数、`Waiting` 恒为 0**（请求压根没进引擎的等待队列，不是引擎调度不过来）；③ **短 bench 方差大到没法比较配置**（20 条 prompt 三次极差 6.7%，加到 200+ 条降到 0.16% 且绝对值高 30%）。最坑的次生现象是**改对了反而测出变慢**：forward step 变快 → 请求更快做完 → 前端更喂不上 → batch 更小、空转更多 → E2E 退步（实测一个 trace 里确认单 step 快 16.5% 的改动，在短 bench 上测出 −7.6%）。
- **触发场景**：充要条件是「prompt 以**文本**形式发给 server 且单条很长」——vLLM 自带的 benchmark 数据集都是 `tokenizer.decode(tokens)` 后发文本，server 端要对每条做一次完整 encode（7.5 万 token 量级才显著）；闭环并发 + 短 OSL + 高 prefix cache 命中率会加剧（请求结束越快，对补货速率要求越高）。**发 `prompt_token_ids` 则完全绕开**。注意 `Running` 偏低本身没有判别力（chunked prefill 下一条请求会跨多个 step 留在 running），**真正的信号是 `Waiting` 恒为 0**。
- **根因**：tokenize 确实被 offload 到线程池、不阻塞事件循环，但 `vllm/config/model.py` 的 `renderer_num_workers` **默认是 1**，`vllm/renderers/base.py` 据此建 `ThreadPoolExecutor(max_workers=1)`；HF fast tokenizer 是 Rust 实现、encode 时释放 GIL，所以并行本是有效的（vLLM 还配套给每个 worker 发了 tokenizer 副本），只是默认值保守。于是 N 条并发的 encode 在一个线程里排队，请求进不了 waiting 队列，调度器无料可调。
- **解决方案**：`vllm serve ... --renderer-num-workers 4`，实测吞吐 +11.5%、median TTFT −8.4%，且 `Waiting` 首次出现非零。**无效的直觉做法**：调大 `--max-num-batched-tokens`（治标且真实服务里有害）、调大并发（队列堆在前端，TTFT 翻倍吞吐不涨）。另外**必须同时把 bench 窗口拉到 ≥30s**（按目标 duration 反推条数，不要用 `CONC*2` 这类与窗口无关的公式），否则暂态占比过高、A/B 结论会反。详见 [2_vllm_renderer_num_workers_frontend_bottleneck/](2_vllm_renderer_num_workers_frontend_bottleneck/)。

### 3. sglang MXFP8 scale 改名静默丢掉 NVFP4 expert block scale，输出乱码

- **现象描述**：服务正常起来、日志无任何加载 warning，但输出乱码（复读系统提示、`ai，ai…`），换 attention / KV dtype / MoE kernel 都不变；**注意乱码可以有很多原因，出现此现象不代表就是本条根因**，必须先按详细说明里的「快速判断」确认 expert 的 `weight_scale` 确实没被加载。
- **触发场景**：sglang upstream（实测 rocm/sgl-dev v0.5.21 的 `9f5d871251`）加载 `quant_config.get_name()=="mxfp8"` 的 checkpoint，且其中有本来就叫 `.weight_scale` 但不是 MXFP8 scale 的张量——实测为 MiniMax-M3.1（MXFP8 linear + compressed-tensors NVFP4 experts，hybrid 配置继承 Fp8Config）；v0.5.17 无此逻辑不触发。
- **根因**：`DefaultModelLoader.load_weights_only` 在 quant name 为 `mxfp8` 时把所有 `*.weight_scale` 改名为 `*.weight_scale_inv`，NVFP4 expert block scale 被改成不存在的参数名，而 model 的 expert 加载分支找不到参数时**静默 return**，scale 保持 `torch.empty`（各 TP rank 值相同）。
- **解决方案**：hybrid 配置跳过该改名；排查方法（逐层 dump 对比 + loader 打点看哪些张量没到达）和走过的弯路见 [3_sglang_mxfp8_scale_rename_drops_nvfp4_expert_scales/](3_sglang_mxfp8_scale_rename_drops_nvfp4_expert_scales/)。

### 4. sglang PD（mooncake）长时间压测后临时端口耗尽，prefill 报 `Cannot assign requested address`

- **现象描述**：PD 精度测试都正常，长 prompt 持续压十几分钟后 prefill 日志刷屏 `TcpTransport::getConnection failed to create connection to <本机IP>:<端口>. Error: connect: Cannot assign requested address`，客户端只看到 router 的 `No available prefill workers (all circuits open or unhealthy)`；宿主机 TIME_WAIT 成千上万，几乎全连向 decode 各 TP rank 的 mooncake 端口，压测期间每 25 s 翻倍。
- **触发场景**：mooncake 后端、intra-node 无 RDMA（实测 MI355X 单机 PP4 -> TP4，75k prompt 并发 4~16）且请求持续不断；只有 host 内存里的 aux 元数据走 mooncake 的 TCP 通道（KV 等显存数据走 HIP 不受影响），实测单个请求在 4 个 decode 端口上留下约 150 个 TIME_WAIT。
- **根因**：mooncake TCP transport 默认不开连接池（`enable_connection_pool_=false`），每个 slice 新建一条连接、传完即关，TIME_WAIT 60 s 才释放；连的是本机网卡 IP 而非 loopback，`tcp_tw_reuse=2` 不生效，持续负载下单个目标端口的约 2.8 万个源端口被用光。
- **解决方案**：prefill 和 decode 都 `export MC_TCP_ENABLE_CONNECTION_POOL=1`，实测 75k 并发 32 端口报错为 0；它常和 PD staging 的 300 s 卡顿同时出现但不是一回事，排查脚本和弯路见 [4_mooncake_tcp_aux_ephemeral_port_exhaustion/](4_mooncake_tcp_aux_ephemeral_port_exhaustion/)。

### 5. ROCm 上 sglang 的 tokenizer worker 在第一张卡上建 HIP context，KFD runlist 过订阅，scheduler kernel 周期性挂起 ~12 ms

- **现象描述**：多模态模型开 `--tokenizer-worker-num` > 1 时，吞吐整体掉一截（实测 tw=8 从约 200k 掉到 107k–175k/卡，tw=16 约为正常的 1/26），同一配置每次重启结果不同、且整个服务生命周期一直慢（不只是 warmup 期间）；`rocm-smi` 里只有第一张可见卡长期 100% 忙、其余卡大半时间空等，第一张卡上有几个 `sglang::tokenizer_worker` 各占约 682 MB 显存却从不计算；dmesg 有 `amdgpu: Runlist is getting oversubscribed due to too many queues. Expect reduced ROCm performance.`；torch trace 里只有第一张卡的 rank 上任意 kernel 偶发被拉长约 12 ms（量子化在 11–14.5 ms）。
- **触发场景**：ROCm（KFD HWS 模式；MI300/MI355 每个 XCC 只有 24 个用户 compute HQD）上，**有别的进程比服务进程晚**在同一张卡上建 HIP 队列，且这张卡上的 compute queue 总数超过 24——实测的来源是 sglang 多模态模型 + `--tokenizer-worker-num>1`：启动 warmup 每个 worker 发一条带图请求，处理到图的 worker 在 `cuda:{base_gpu_id}` 上跑 fast image processor，各建 2 个 compute queue，而 scheduler 在 `GPU_MAX_HW_QUEUES=16` 下自己就占 17 个。反直觉的边界：**比服务进程早建的空闲队列再多也没事**（实测 25、29 个都 0 长尾），晚建的哪怕完全空闲、不提交任何工作也会触发；有几个 worker 建 context 取决于 warmup 请求落在哪，是随机的（tw=4 可能中也可能不中）；另一个反直觉点：warmup 只发一次图，但建出的 context 和队列在 worker 进程退出前不会释放，所以 server 就绪后的压测照样中招。
- **根因**：KFD 构建 HWS runlist 时新进程排在前面，HWS 按顺序把队列映射到 24 个 HQD；总数超过 24 时 runlist 过订阅、HWS 时间片轮转，放不下的是排在最后的 scheduler 的队列，于是 scheduler 正在跑的 kernel 每轮被换下约一个 quantum（KFD 设 `process_quantum=10`，实测 ≈12 ms；单位和固件策略属推测）。
- **解决方案**：让 tokenizer 进程不碰 GPU：在 ROCm 上把 tokenizer 侧的 fast image processor 放到 CPU（`SGLANG_ROCM_MM_PREPROCESS_ON_GPU=1` 可恢复），实测 MiniMax-M3.1-Flash PP4 prefill 每卡从 107k 到约 200k；没有补丁时只能把 tw 降到 ≤3（多模态流量一来仍可能越界）。快速判断命令和控制实验见 [5_rocm_kfd_runlist_newer_process_starves_scheduler/](5_rocm_kfd_runlist_newer_process_starves_scheduler/)。

## 文件结构

```
trouble_shooting/
├── README.md                     # 本文件：目录 + 每条的四段式概述 + 格式要求
└── <序号>_<英文slug>/
    ├── README.md                 # 详细根因、完整调查过程、修复方案
    └── <复现脚本 / 补丁 / 日志>    # 让别人能自己复现，而不只是读结论
```

序号从 1 递增，slug 用小写英文加下划线，能一眼看出是什么问题。

## 格式要求

### 本文件里的条目

每条固定四段，**各用一句话**说清楚：

| 段 | 内容 |
|---|---|
| 现象描述 | 看到的表象，尽量带上可被搜索到的原始报错/日志特征 |
| 触发场景 | 什么情况下会踩中；如果有反直觉的边界（比如"设置者自身不受影响"）就一并点出 |
| 根因 | 机制，不要停在"改了 X 就好了" |
| 解决方案 | 简单的就直接写完；复杂的写一句话方向 + 指向子目录的详细说明路径 |

### 子目录里的详细说明

`<序号>_<slug>/README.md` 至少覆盖：现象、快速判断（一段能立刻确认"是不是这个坑"的命令或代码）、
触发条件（**充要条件**，以及哪些维度**无关**——这同样重要，能省掉别人的排查时间）、
根因、修复（含**无效的直觉做法**及其原因）、影响范围、复现方式。

### 通用要求

0. **Ask before write**。在往本指南添加内容之前，请询问user，得到同意后才允许写入。
1. **区分实测与推测**。写"在 X 上实测"还是"按机制推测应该也会"，不要混为一谈。
2. **记下走过的弯路**。哪条线索是误导性的（比如某个检查显示"无变化"其实是工具本身看不见），对下一个人的价值往往比正确答案更大。
