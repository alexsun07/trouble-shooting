# vLLM 前端 tokenize 单线程，把引擎饿住

长 prompt 的 serving bench 里，GPU 看着很忙、吞吐却上不去，而且**短 bench 的方差大到没法比较配置**。
根因是 vLLM 的 renderer 线程池默认只有 1 个 worker，前端 tokenize 喂不满引擎。

- **实测环境**：MI355X ×8 / gfx950 / ROCm 7.2.4，`rocm/vllm-dev:nightly_cdna4_main_torch__0913_b238`
  （vLLM `0.9.2rc2.dev13685+g38432c905`，来自 ROCm/vllm），MiniMax-M3-MXFP4 / TP4
- **负载**：ISL=75000（90% 共享前缀）/ OSL=1 / prefix caching 开
- 同类问题在 **sglang** 上也遇到过，对应参数是 `--tokenizer-num-worker`（默认同样是 1，
  改成 4 之后"性能稳定且大幅提升"）。**vLLM 侧是本文实测，sglang 侧是先验经验。**

## 现象

三个信号，单看每一个都不像 tokenizer 的问题：

1. **吞吐上不去，但 GPU 利用率很高**。torch trace 里 GPU busy 95~98%，
   kernel 本身也没有异常慢的。
2. **`Running` 偏低、`Waiting` 恒为 0**。服务日志里那行
   `Engine ... Running: N reqs, Waiting: M reqs`——
   打 `--max-concurrency 10` 的 bench，`Running` 只有 1~5，`Waiting` **从来不是 0 以外的值**。
3. **短 bench 方差爆炸**。同一个服务同一个配置，20 条 prompt 测三次
   极差 6.7%；把条数加到 200+，极差降到 0.16%，而且绝对值高 30%。

第 2 条是最有判别力的：**`Waiting` 恒为 0 说明请求根本没进到引擎的等待队列**，
不是引擎调度不过来。如果是 GPU 侧瓶颈，队列应该会堆起来。

还有一个更隐蔽的次生现象：**优化做对了反而测出变慢**。
forward step 变快 → 请求更快做完 → 前端更喂不上 → batch 更小、step 间空转更多
→ E2E 反而退步。我们就在这里被坑过：一个 trace 里确认单 step 快了 16.5% 的改动，
在短 bench 上测出 −7.6%。

## 快速判断

```bash
# 1. 看队列深度。Running 低于并发数、Waiting 恒为 0 → 基本可以确诊
grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs" <serve.log> | sort | uniq -c | sort -rn

# 2. 看当前配置（默认 1）
grep -oE "renderer_num_workers[^,]*" <serve.log>

# 3. 确认 bench 发的是文本而不是 token id（发 token id 就不会有这个问题）
#    vLLM 自带的 random / prefix_repetition 等数据集都是发文本
```

实测前后对比（同一台机器、同一模型配置，只改这一个参数）：

| | Running / Waiting 分布 | throughput | median TTFT |
|---|---|---:|---:|
| `renderer_num_workers=1` | `2/0` ×20、`1/0` ×17、`3/0` ×2 —— **Waiting 全 0** | 455,238 | 1,600 ms |
| `renderer_num_workers=4` | **`5/6`** ×7、`3/0` ×5、`4/0` ×2 | **507,422** | **1,465 ms** |
| | | **+11.5%** | **−8.4%** |

`Waiting` 第一次出现非零，就是瓶颈从前端移回 GPU 的标志。
注意**吞吐和延迟同时改善**——这是解掉瓶颈的特征，不是拿延迟换吞吐。

## 触发条件

**充要条件是"prompt 以文本形式发给 server，且单条 prompt 很长"。**

- **必要**：客户端发文本。vLLM 的 benchmark 数据集
  （`vllm/benchmarks/datasets/datasets.py`）普遍是
  `prompt = tokenizer.decode(tokens)` → `SampleRequest(prompt=prompt)`，
  `endpoint_request_func.py` 再 `payload["prompt"] = ...` 发出去。
  server 端因此要对每条请求做一次完整 encode。
  如果走 `prompt_token_ids` 直接发 id，就绕开了整条路径。
- **必要**：prompt 足够长。7.5 万 token 的 encode 才会显著到能卡住补货；
  几百 token 的请求不会。
- **加剧**：闭环并发（`--max-concurrency`）+ 短 OSL。
  OSL=1 时请求做完 prefill 立刻结束，补货速率直接决定 batch 大小。
- **加剧**：prefix caching 命中率高。命中率越高，单条请求实算越少、结束越快，
  对补货速率的要求越高。

**无关的维度**（排查时可以直接跳过）：

- 不是 `--max-num-batched-tokens` 的问题。把它调大只是迁就 benchmark 的规律性，
  真实服务里抬 TTFT、纯亏。
- 不是调度器不肯填满 batch。`vllm/v1/core/sched/scheduler.py` 里
  `while (self.waiting or self.skipped_waiting) and token_budget > 0:`——
  只要 waiting 非空、预算没用完，它就会继续切一块进来。**问题在 waiting 是空的。**
- 不是 GPU / kernel 慢。trace 里 GPU busy 95~98%。

## 根因

tokenize 确实被 offload 到线程池了（不阻塞事件循环），但**池子只有一个线程**。

`vllm/config/model.py`：

```python
renderer_num_workers: int = 1
"""Number of worker threads in the renderer thread pool. The pool is
consumed by the async renderer path (e.g. the OpenAI-compatible API
server started by `vllm serve`) to parallelize tokenization, chat
template rendering, and multimodal preprocessing across concurrent
requests."""
```

`vllm/renderers/base.py`：

```python
pool_workers = config.model_config.renderer_num_workers
self._executor = ThreadPoolExecutor(max_workers=pool_workers)
...
# Offload tokenization to the thread pool.
self._tokenize_prompt_async = make_async(self._tokenize_prompt, executor=self._executor)
```

HF fast tokenizer 是 Rust 实现、encode 时会释放 GIL，所以**加线程是真能并行的**。
`vllm/renderers/hf.py` 里配套给每个 worker 发了一份 tokenizer 副本
（`maybe_make_thread_pool(self.tokenizer, renderer_num_workers + 1)`），
因为 HF tokenizer 不是线程安全的——这个设施存在本身就说明并行是设计内的，
只是默认值保守。

于是：N 条并发请求的 encode 在一个线程里排队 → 请求进不了引擎的 waiting 队列
→ 调度器无料可调 → batch 偏小 + step 间空转。

## 修复

```bash
vllm serve ... --renderer-num-workers 4
```

**无效或误导的做法**：

- `--max-num-batched-tokens` 调大：治标，且真实服务里有害。
- 加大 `--max-concurrency`：队列堆在客户端/前端，TTFT 翻倍，吞吐几乎不涨。
- 加大 bench 的 `--num-prompts`：能让数字变好看（拉长窗口摊薄暂态），
  但没解决瓶颈。**不过这条本身是必须做的**，见下。

**相关但不同的旋钮**：`--api-server-count`（非 DP 场景默认 1）扩的是整个
API server 进程，不只是 tokenize 线程池。如果瓶颈在 HTTP/序列化/响应处理而不是
encode，那个才管用。**本文没有实测它。**

**4 是不是最优没测过**，只实测了 1 → 4。

## 附带的坑：bench 窗口太短会放大这个问题

这个瓶颈会让**短 bench 完全不可用**，因为暂态占比太高：

| NUM_PROMPTS | duration | throughput | 极差 |
|---:|---:|---:|---:|
| 20 | ~4s | 353,085 (n=3) | **6.71%** |
| 200 | 32–33s | 444,031 / 461,840 | |
| 223 | ~36s | 466,672 / 467,428 | **0.16%** |
| 500 | 82s | 455,238 | |

同一个服务同一个配置，只改条数就差 32%。200 以上进入平台期，500 没有更好。

**所以定 bench 条数要按目标 duration 反推**（≥30s），不要用
`CONC * 2` 这类和窗口长度无关的公式。我们就是因为默认 `NUM_PROMPTS=CONC*2=20`，
把两个优化项的收益判断全搞反了。

## 我们走过的弯路

1. **先怀疑 `--max-num-batched-tokens`**。trace 里看到 step 序列是
   `32768, 4952, 32768, 4952...`，那个 4952 的尾巴效率只有大 step 的 66%，
   于是以为是 token 预算卡在尴尬的值上。查了调度器源码才确认它会主动 backfill，
   问题在"没东西可 backfill"。
2. **误读 `Running: 4-5` 为"正常"**。chunked prefill 下一条请求会在 running 里
   待好几个 step，这个数确实不等于"引擎里有几条"。**真正有判别力的是 `Waiting`。**
3. **用 4 个日志采样点下结论**。服务日志里 `Running/Waiting` 是 ~10s 一行，
   而 bench 窗口只有 30~80s。样本太少。要看清楚应该按 0.25s 轮询 `/metrics` 的
   `vllm:num_requests_running` / `vllm:num_requests_waiting`。
4. **在短 bench 上做 A/B，把对的改动判成错的**。见上面「现象」第 4 段。

## 一个容易混淆的相邻问题

sglang 的 GSP 数据集有 decode→encode round-trip 不守恒的问题
（实测 67500 token → 69069，虚高 2.3%，见
`tasks/minimax-m3/NOTES_metrics_and_gsp.md`）。
**vLLM 这边不存在**——`PrefixRepetitionRandomDataset` 用
`gen_prompt_decode_to_target_len()` 做了补偿，实测 `total_input_tokens`
精确等于名义值、命中率稳定在理论天花板。两个问题都出在"客户端造 prompt"这一环，
但不是一回事，别混。
