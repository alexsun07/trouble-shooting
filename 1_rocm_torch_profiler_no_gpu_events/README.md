# ROCm 上 torch profiler 抓不到任何 GPU kernel

**一句话**：进程启动时环境里带着 `ROCPROFILER_REGISTER_LIBRARY`，它的 torch profiler
就抓不到任何 device 事件。这个变量由 rocprofiler 在**第一次触碰 HIP 时用 C 层 `setenv()`
注入**，会被所有后代进程继承，而且**对 Python 的 `os.environ` 不可见**。

调查于 2026-08-20，MI355X / gfx950，torch 2.13.0+rocm7.14.0。

---

## 0. 典型案例（先看这里对号入座）

这个坑最麻烦的地方是：**它几乎从不以"profiler 坏了"的形式出现**，而是伪装成别的问题。
下面两类是实测踩到过的，任何"父进程查一下 GPU → spawn 出 worker 干活"的工具都可能中。

| 场景 | 你看到的 | 你会以为是 | 实际是 |
|---|---|---|---|
| **kernel tuning**（aiter `gemm_moe_tune.py` 等） | 所有候选 `us=0`，最后 `tune 0 shapes` / `no valid candidate found` | kernel 不支持这个 shape / 数据构造错了 / 多进程有问题 | kernel 跑得好好的，只是**计时用的 profiler 一个事件都没采到** |
| **采集 profile trace**（sglang `/start_profile` 等） | trace 文件正常生成、几十 MB、无任何报错 | 抓成功了 | 文件里**只有 CPU 事件**，`kernel` 类别为 0 |

### 案例 A：aiter MoE tuning，所有候选 `us=0`

跑 `gemm_moe_tune.py` tune MoE kernel 时，448 个 FlyDSL 候选**全部** `us=0`，
日志刷屏 `no valida data after post process!`，最终 `tune 0 shapes`。

**为什么会被带偏**：报错文字指向的是 errRatio 和候选有效性
（`Error: please check errRatio, stage1 and stage2 should be valid together!`），
把人往"这个 shape 不被支持""activation 分支写错了""多进程调度有问题"上引。
我们当时就在这上面绕了很久，甚至一度记录成"Swiglu 下全 0、换 Silu 就正常"——
换台机器复现时 **Silu 一样全废**，那个"信号"根本是噪声。

**一句话判据**：拿 `torch.cuda.Event` 给同一个候选计一次时，或者干脆在**主进程**里
直接跑同一个 kernel。我们实测 Event 是 860–950 µs、主进程 profiler 是 827 µs——
**kernel 完全正常，空的是 profiler**。只要这两个数对得上，就可以立刻排除"kernel 有问题"。

**当时的解法**（见第 5 节"替代方案"）：没有定位到根因，而是改用 `torch.cuda.Event`
计时绕开 profiler，靠这个产出了完整的 tuned config。
根因是后来查采集 trace 的问题时才挖出来的。

**两条路都实测有效，可任选**。定位到根因之后回过头验证：只挂 sitecustomize、
**完全不动 aiter 代码**（补丁是 opt-in 的，不设 `AITER_USE_CUDA_EVENT` 就不生效），
448 个候选全部正常计时，`no valida data` / `us = 0` 均为 0 行，正常产出 tuned 行：

| | profiler 路径（仅 sitecustomize） | cuda.Event 路径（aiter 补丁） |
|---|---|---|
| 选中的 stage1 kernel | `flydsl_moe1_..._t128x128x256_w2_bnt0_gui_fp8` | **同一个** |
| us1 / us2 | 389.71 / 217.33 | 418.50 / 237.37 |
| 总计 | **607.03 µs** | 655.86 µs |
| TFLOPS | 955.17 | 884.06 |

两条路**选出同一个 stage1 kernel**，说明排序结论一致；profiler 读数系统性低约 7%，
正好符合预期——cuda.Event 测的是含 launch 开销的 wall time，profiler 测的是纯 device
time，**二者的差值方向和量级都合理**，互相印证。

> 验证时的坑：第一次验证只看了"`no valida data` 行数变成 0"就以为成功，
> 其实那次容器已经挂了（exit 137）、tuner 根本没跑起来——**0 行是空跑的产物**。
> 判据必须是**正向证据**：确认候选数（`Distributing 448 task groups`）、
> 确认真的产出了 tuned 行，而不是只看某个错误计数归零。

### 案例 B：sglang 采集 profile trace，trace 里没有 GPU

`--ep 4` 起的 server 调 `/start_profile` 抓 trace，4 个 rank 各产出 46–105 MB 的
`.trace.json.gz`，看起来一切正常，实际打开一看 `kernel` 事件数为 0，
只有 `python_function` / `cpu_op`。

**为什么会被带偏**：整个流程**没有任何报错**，文件大小也正常——不去数事件类别根本发现不了，
很容易就拿着一份没有 GPU 的 trace 去做性能分析。

**一句话判据**：见下面第 1 节的 `event cats` 对比，有没有 `kernel` 一眼就能看出来。

---

## 1. 现象

同一个根因会以两种完全不同的面目出现。

**A. profile 出来的 trace 没有 GPU**

trace 能正常生成、大小也正常（几十 MB），但事件类别里**没有 `kernel`**：

```
event cats: {'python_function': 1959758, 'cpu_op': 60887, 'user_annotation': 24656, 'Trace': 1}
device events: 0
```

正常应该长这样（注意多出来的 `kernel` / `cuda_runtime` / `ac2g`）：

```
event cats: {'python_function': 1959758, 'cpu_op': 60887, 'ac2g': 5517,
             'cuda_runtime': 3269, 'kernel': 2247, ...}
device events: 2248
```

**B. 基于 profiler 计时的工具全部返回 0**

aiter 的 MoE tuner 就是典型：每个候选都是 `us=0`，日志刷屏

```
[aiter] no valida data after post process!
!!!! us = 0, try 1 run
!!!! us = 0, try 2 run
!!!! us = 0, try 3 run
Warning: try run 3 times, but still get 0!
...
Error: please check errRatio, stage1 and stage2 should be valid together!
no valid candidate found for (...)
[aiter] Tuning Error. tune 0 shapes
```

链路：`aiter/test_common.py::post_process_data()` 第一步
`df[df["device_type"].str.contains("DeviceType.CUDA")]` 拿到空表 → `return [], 0`
→ `us=0` → 重试 3 次仍为 0 → 候选作废。

> ⚠️ 这类工具的报错会把人往"kernel 不支持 / 数据构造错了 / 多进程有问题"上引。
> **kernel 其实跑得好好的**——我们用 `torch.cuda.Event` 实测同一个候选是 860–950 µs，
> 与主进程里 profiler 报的 827 µs 一致。空的是 profiler，不是 kernel。

## 2. 快速判断

在**出问题的那个进程里**（不是它的父进程）跑：

```python
import ctypes
libc = ctypes.CDLL("libc.so.6"); libc.getenv.restype = ctypes.c_char_p
print(libc.getenv(b"ROCPROFILER_REGISTER_LIBRARY"))   # 非空 => 中招
```

必须用 `ctypes` 读 C 层 `getenv`。**`os.environ` 在设置它的那个进程里看不到**
（子进程里能看到，因为是 exec 时继承的）。

最小复现（30 秒，不依赖任何框架）：

```bash
LIB=$(python3 -c "import _rocm_sdk_core,os;print(os.path.dirname(_rocm_sdk_core.__file__))")/lib/librocprofiler-sdk.so.1
python3 probe_profiler_trigger.py profile                          # 期望 SEES
env ROCPROFILER_REGISTER_LIBRARY=$LIB python3 probe_profiler_trigger.py profile   # 期望 BLIND
```

## 3. 触发条件

### 充要条件

> **进程启动时，环境变量 `ROCPROFILER_REGISTER_LIBRARY` 已被设置为非空、且可以 dlopen
> 的路径。**

跟父子关系、进程树形态**都没有关系**——父子只是这个变量最常见的传播途径。实测：完全没有
父进程参与，只是 `env ROCPROFILER_REGISTER_LIBRARY=<真实路径> python3 xxx`，单个进程就
BLIND。

### 变量是怎么被设上的

它由 rocprofiler 在**第一次触碰 HIP 运行时**注入。实测各阶段（`ctypes.getenv` 观测）：

| 阶段 | C environ | os.environ |
|---|---|---|
| 进程启动 | unset | unset |
| `import torch` | unset | unset |
| **`torch.cuda.is_available()`** | **SET** | unset |
| `torch.cuda.device_count()` | SET | unset |
| `torch.cuda.set_device(0)` | SET | unset |
| 分配显存 / 跑第一个 kernel | SET | unset |

**`torch.cuda.is_available()` 就足够了**——不需要分配显存，也不需要跑任何 kernel。
这就是为什么各种"先检查一下有没有 GPU"的 launcher 会静默地毁掉所有 worker 的 profile。

注意 `os.environ` **全程都是 unset**。

### 会踩中的场景

1. **多进程 launcher**：父进程检查/查询 GPU，再 spawn worker。
   实测中招的：**sglang**（scheduler rank 全是 CPU-only trace）、
   **aiter MoE tuner**（worker 全部 `us=0`）。
   按同样机制，`torch.multiprocessing` spawn 的训练/推理 worker 也会中。
2. **经过任意中间进程**：父 → `bash -c` → python，照样 BLIND（实测）。
3. **CI / pytest**：先跑了一个用 GPU 的用例，再 `subprocess` 起需要 profile 的任务。
4. 手动 `export` 了这个变量（罕见）。

### 不会触发

| 情况 | 结果 |
|---|---|
| 单进程自己 init HIP 再 profile（**设置者自身免疫**） | SEES ✅ |
| 父进程从不碰 HIP，子进程 profile | SEES ✅ |
| 显式传 `env=<dict>` 且不含该变量（如 `subprocess.run(env=...)`）| SEES ✅ |
| 变量存在但值为**空字符串** | SEES ✅ |

"设置者自身免疫、后代中招"这个不对称是本坑最反直觉的地方：**在父进程里怎么测都是好的**。

### 与之无关的维度（已逐一排除，别再查了）

- **不是 fork 的问题**：`mp_tuner` 本来就 `set_start_method("spawn")`，spawn 一样中招。
- **不是 profiler session 冲突**：父进程**从不** profile 也照样复现。
- **不是并发互斥**：另起一个独立进程（非父子）持有 HIP 上下文、甚至同时有一个占满
  4 张卡的 sglang server 在跑，单独 profile 依然正常（SEES 10）。
- **不是 per-device**：父进程只碰 GPU0，子进程 profile GPU1，照样 BLIND。
- **与是否同一张卡、父进程是否在 profile 无关。**

## 4. 根因

`librocprofiler-sdk` 通过 `rocprofiler-register` 机制在 HSA 运行时初始化时登记自己，
登记的同时用 C 的 `setenv()` 把自己的路径写进环境：

```
ROCPROFILER_REGISTER_LIBRARY = <...>/librocprofiler-sdk.so.1
HSA_TOOLS_ROCPROFILER_V1_TOOLS = 0
GLOG_minloglevel / GLOG_logtostderr / GLOG_alsologtostderr / GLOG_stderrthreshold
```

新进程启动时如果这个变量**已经存在**，rocprofiler-register 会**强制把它指向的库
dlopen 进来、装成本进程的 profiling tool**。ROCm 的 tracing 后端是单消费者，
rocprofiler-sdk 占住 API table 拦截之后，PyTorch 的 kineto 就拿不到 dispatch 回调，
profiler 里一个 device 事件都没有——**等价于每个后代进程都被隐式套上了一层
`rocprofv3`**（正是第 7 节那个"两者不能同时用"的坑，同一个机制）。

代码在 `rocprofiler-register/source/lib/rocprofiler-register/rocprofiler_register.cpp`
的 `rocp_reg_scan_for_tools()`：

```cpp
auto _rocp_reg_lib = get_env("ROCPROFILER_REGISTER_LIBRARY", "");
bool _force_tool   = get_env("ROCPROFILER_REGISTER_FORCE_LOAD",
                             !_rocp_reg_lib.empty() || !_rocp_tool_libs.empty());
bool _found_tool   = (rocprofiler_configure || _configure_func || _force_tool);

if (_found_tool) {
    if (rocprofiler_lib_handle && rocprofiler_lib_config_fn)  // 进程内静态缓存
        return rocp_scan_data{...};                           // 与环境变量无关
    if (_rocp_reg_lib.empty()) _rocp_reg_lib = rocprofiler_lib_name;
    std::tie(handle, config_fn) = rocp_load_rocprofiler_lib(_rocp_reg_lib);  // dlopen 并挂上
}
```

关键在于 `_force_tool` 的**默认值就是 `!_rocp_reg_lib.empty()`**：变量非空 ⇒
`_force_tool` 为真 ⇒ `_found_tool` 为真 ⇒ 走进 dlopen 分支。函数开头那处 `return`
是**进程内的静态缓存**，只在同一进程第二次调用时生效，与环境变量无关。

这也把本坑最反直觉的不对称解释清楚了：

| | 进程启动时的变量 | `_force_tool` | sdk 是否被挂成 tool | kineto |
|---|---|---|---|---|
| 设置者（父） | 空 | false | 否 | **SEES** ✅ |
| 继承者（子） | 非空 | **true** | **是** | **BLIND** ❌ |

第 5 节的 `sitecustomize` 之所以有效，就是把子进程还原成"启动时变量为空"的状态。

实测逐个排除，六个泄漏变量里**只有 `ROCPROFILER_REGISTER_LIBRARY` 是真凶**：

| 环境 | 结果 |
|---|---|
| 完整泄漏环境 | blind（复现）|
| minus `GLOG_*`（4 个逐个试） | blind |
| minus `HSA_TOOLS_ROCPROFILER_V1_TOOLS` | blind |
| **minus `ROCPROFILER_REGISTER_LIBRARY`** | **SEES** ✅ |

补充：值必须是**可加载的库**。指向一个不存在的路径会让 HIP 初始化直接失败
（`RuntimeError: CUDA unknown error`），空字符串则无害。

### 为什么 `os.environ` 看不到

Python 的 `os.environ` 是**解释器启动时对 `environ` 的一次性快照**，之后 C 代码调
`setenv()` 不会同步回去。所以：

- 在**设置者**进程里：`os.environ` 看不到 → `os.environ.pop()` 无效，只能 `os.unsetenv()`；
- 在**子进程**里：变量是 exec 时随 `environ` 继承的，Python 启动时就读到了 →
  `os.environ` **能**看到，`pop()` 和 `os.unsetenv()` 都有效（两种都实测过）。

## 5. 修复

在**中招的那个进程里、且早于它自己的 HIP 初始化**清掉这个变量。`sitecustomize.py`
会在解释器启动时自动 import，正好在这个时机：

```python
# sitecustomize.py
import os
os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")
```

```bash
PYTHONPATH=/path/to/dir <原来的命令>
```

本目录的 [`sitecustomize.py`](sitecustomize.py) 可直接用。实测效果：
同一条抓取路径 `kernel: 0` → `kernel: 2247`。

### 无效的做法（别浪费时间）

| 做法 | 结果 | 为什么 |
|---|---|---|
| shell 里 `unset ROCPROFILER_REGISTER_LIBRARY` | **BLIND** | shell 阶段变量**还不存在**（实测 `<UNSET>`），它是后来 python 在 HIP init 时才注入的，`unset` 一个不存在的变量是空操作 |
| shell 里 `export ROCPROFILER_REGISTER_LIBRARY=` | **BLIND** | 会被 rocprofiler 的 `setenv(..., overwrite=1)` 覆盖 |
| 在父进程里 `os.environ.pop(...)` | 无效 | 父进程的 `os.environ` 里根本没有这个 key |

### 替代方案：绕开 profiler 计时

如果目标只是**计时**（而不是要一份 trace），可以不修 profiler，直接改用
`torch.cuda.Event`——它在子进程里一直是正常的。

> 对 aiter tuning 这个场景，**上面的 sitecustomize 已经够用**（实测见第 0 节案例 A，
> 两条路选出同一个 kernel）。下面这条路的价值在于**不依赖 profiler**：
> 万一 rocprofiler 的行为在别的版本上又变了，它仍然成立。

aiter 的 `perftest` 自带 `use_cuda_event` 参数，只是没有哪个 tuner 传过它，
也没有环境变量开关；加两行让它认一个环境变量即可（默认行为不变）：

```python
_use_evt = use_cuda_event or int(os.environ.get("AITER_USE_CUDA_EVENT", "0"))
if int(os.environ.get("AITER_LOG_MORE", "0")) or _use_evt:
    ...
    if _use_evt:
        return data, avg
```

代价：cuda.Event 测的是每次调用的 wall time（含 launch 开销），不是 profiler 的纯
device time。对几百 µs 以上的 kernel 可忽略；很小的 kernel 会有噪声，但所有候选同等
受影响，排序基本保留。

**这条路是实测有效的**：同一条最小复现命令，`us=0` 的候选从 4/4 变成 0/4
（实测值 857 / 850 / 930 / 905 µs），并据此完成了 16 档 shape 的完整 tuning。
它不依赖 profiler，因此比 sitecustomize 那条路更稳——即使将来 rocprofiler 行为再变也不受影响。

## 6. 影响范围

**实测环境**（以下都是在这台机器上跑出来的）：

| 项 | 值 |
|---|---|
| 硬件 | MI355X (gfx950) |
| torch | 2.13.0+rocm7.14.0，HIP 7.14.60850 |
| ROCm 形态 | **pip 安装**（`rocm-sdk` / `_rocm_sdk_core`），容器内**没有** `/opt/rocm` |
| 泄漏值 | `<site-packages>/_rocm_sdk_core/lib/librocprofiler-sdk.so.1` |
| 镜像 | `rocm/atom-dev:sglang-v0.5.15.post1-nightly_20260811` |
| 实测中招的软件 | sglang 0.5.15.post1、aiter（`gemm_moe_tune.py` / `mp_tuner`） |

第二次复现（2026-08-26）：**MI308X (gfx942)**，同一镜像、torch 2.13.0+rocm7.14.0，
sglang TP4 的 4 个 scheduler rank 全是 CPU-only trace；挂上 `sitecustomize` 后
同一条采集路径 `kernel: 0` → `kernel: 11370`（`ac2g` / `cuda_runtime` /
`gpu_user_annotation` / `gpu_memcpy` 也一并出现）。说明与具体 GPU 型号无关。

### 版本门控：只有新版 rocprofiler-sdk 会注入

2026-08-26 在 MI308X (gfx942) 上做了跨容器对照，同一台机器四个容器：

| 容器 ROCm | 形态 | `librocprofiler-sdk` 里引用该变量 | HIP init 后是否注入 | spawn 子进程 kineto |
|---|---|---|---|---|
| **7.14** | pip (`_rocm_sdk_*`) | **4 处** | **注入** | **kernel=0** ❌ |
| **7.14** | pip (`_rocm_sdk_*`) | **4 处** | **注入** | ❌ |
| 7.2.4 | 系统 `/opt/rocm` | 0 处 | 不注入 | kernel=5 ✅ |
| 7.2.3 | 系统 `/opt/rocm` | 0 处 | 不注入 | ✅ |

静态判据（不碰 GPU，可以安全地对别人正在跑的容器执行）：

```bash
strings -a $(find / -xdev -name 'librocprofiler-sdk.so*' | head -1) \
  | grep -c ROCPROFILER_REGISTER_LIBRARY      # 0 = 不会中招；非 0 = 有风险
```

注意**变的是写入端，不是读取端**：`librocprofiler-register.so` 在所有版本里都一样
（都引用该变量、都 import `setenv`），一直支持这个协议；是新版 `librocprofiler-sdk`
才开始在自我登记时把自己的路径回写进环境。

**推测但未验证**（不要当成结论）：

- 非 Python 的 HSA 程序（C++ 直接调 HIP + kineto）按机制推测同样受影响，未测。
- **NVIDIA 不受影响**——这是 rocprofiler 特有的机制。

**未能验证的**：

- `fork` 出来的子进程。fork 一个多线程且已初始化 HIP 的进程本身就不受支持，
  实测子进程无任何输出（疑似静默崩溃）。按"环境变量"这条规则推测它也会 BLIND，
  但没有直接证据。

**上游状态**：用 `gh` 搜过 ROCm/ROCm、ROCm/rocprofiler-register，
没有找到对应的 issue 或 PR（截至 2026-08-20）。

## 7. 附带发现：rocprofv3 和 kineto 不能同时用

用 `rocprofv3 --kernel-trace -- python3 xxx` 包起来跑的时候，**连主进程**的 torch
profiler 也变成 0 事件（正常情况下主进程一直是好的）。ROCm 的 tracing 后端是单消费者，
rocprofv3 占住之后 kineto 就什么都拿不到。

所以想用 rocprofv3 补位的时候，**不要**同时期待 torch profiler 的输出。

## 8. 复现

| 脚本 | 用途 |
|---|---|
| [`probe_profiler_trigger.py`](probe_profiler_trigger.py) | 触发条件全套：`T2` 注入时机、`T7` launcher 模式、`T8` 设置者自身、`T3` fork、`T4` 经 bash 中转 |
| [`probe_profiler_rootcause.py`](probe_profiler_rootcause.py) | 根因定位：`cleanenv` 继承 vs 清洗、`envleak` dump 子进程实际收到的环境、`pinpoint` 逐个变量排除 |
| [`probe_profiler_in_child.py`](probe_profiler_in_child.py) | 最小复现：主进程 10 个 CUDA 事件 vs 子进程 0 个 |
| [`sitecustomize.py`](sitecustomize.py) | 修复 |

```bash
# 最小复现
python3 probe_profiler_in_child.py

# 注入时机
python3 probe_profiler_trigger.py T2

# 最常见的现实场景：父进程只调 is_available()
python3 probe_profiler_trigger.py T7

# 定位到具体变量
python3 probe_profiler_rootcause.py pinpoint

# 验证修复
PYTHONPATH=$(pwd) python3 probe_profiler_rootcause.py cleanenv
```

## 9. 排查时走过的弯路

留给下一个人，这几步都是**看起来正确但会把你带偏**的：

1. **"对比 HIP init 前后的 `os.environ`"显示零变化** → 差点把环境变量这条线整个排除掉。
   实际上 `os.environ` 结构性地看不见 C 层 `setenv`。**要用 `ctypes.getenv`。**
2. **逐个删环境变量做 bisect 时，删掉任意一个都能恢复** → 看起来毫无规律。
   真相是：`subprocess.run(env=<dict>)` 会用 Python 的 dict **重建**环境，天然丢掉所有
   C 层注入的变量；而 `env=None` 走 execv 继承**真实的 C environ**。
   差别根本不在删了哪个变量，而在"有没有显式传 `env=`"。
   **正确做法**：从完整的泄漏环境出发，每次只去掉一个，保证唯一变量。
3. **先入为主认为是 fork / 多进程的锅** → `--mp 1` 单进程复现后仍然失败，
   而 mp_tuner 本来用的就是 spawn。
4. **以为是某个 activation / kernel / shape 不支持**（aiter tuner 的场景）→
   曾记录成 "Swiglu 下 448 个候选全部 `us=0`，换 Silu 就正常"。
   换台机器复现时 **Silu 一样全废**，这个"信号"是噪声。
   判据应该是：拿 `torch.cuda.Event` 或直接在主进程里跑同一个 kernel 对照一下，
   立刻就能区分"kernel 没跑"和"profiler 没看见"。
