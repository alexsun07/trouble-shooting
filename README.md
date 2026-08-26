# 防踩坑指南

注意：如果要添加新的记录，请确保已经弄清楚触发场景并定位清楚根因，记录的应该尽可能提供给更general的使用范围而不局限于某一个机器、环境。如果是特有环境才会触发，也应该加以说明。

## 目录

| # | 问题 | 详细说明 |
|---|---|---|
| 1 | ROCm 上 torch profiler 抓不到 GPU kernel | [1_rocm_torch_profiler_no_gpu_events/](1_rocm_torch_profiler_no_gpu_events/) |

---

### 1. ROCm 上 torch profiler 抓不到 GPU kernel

- **现象描述**：这个 bug 几乎从不以"profiler 坏了"的形式出现，而是伪装成别的问题——**两类典型案例**：① **kernel tuning**（aiter `gemm_moe_tune.py`）所有候选 `us=0`、刷屏 `no valida data after post process!`、最终 `tune 0 shapes`，看着像 kernel 不支持或数据构造有问题，实际 kernel 完全正常（`cuda.Event` 实测 860–950 µs）、空的是采集不到 event 的 profiler；② **采集 profile trace**（sglang `/start_profile`）trace 文件正常生成且无任何报错，但里面只有 `python_function` / `cpu_op`、`kernel` 类别为 0。
- **触发场景**：进程启动时环境里已带着 `ROCPROFILER_REGISTER_LIBRARY`——最常见的是父进程调用过 `torch.cuda.is_available()`（不需要跑任何 kernel）之后再起子进程，因此凡是"父进程查一下 GPU → spawn worker 干活"的工具都可能中招（实测：sglang 的 scheduler、aiter tuner 的 worker），而设置该变量的进程**自身不受影响**。
- **根因**：rocprofiler 在第一次触碰 HIP 时用 C 层 `setenv()` 注入该变量，后代进程继承后其注册逻辑判定"已注册"而走短路分支，kineto 收不到 dispatch 回调；该变量对 Python 的 `os.environ` 不可见，所以排查时对比 `os.environ` 会误报"零变化"。
- **解决方案**：用 `sitecustomize.py` 在子进程解释器启动时（早于 HIP init）`os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")` 并挂到 `PYTHONPATH`（可直接取用 [1_rocm_torch_profiler_no_gpu_events/sitecustomize.py](1_rocm_torch_profiler_no_gpu_events/sitecustomize.py)），注意**在 shell 里 `unset` 无效**（那时变量还不存在）——这一条对采集 trace 和 aiter tuning 两个场景都实测有效；若只需计时而不需要 trace，也可以改用 `torch.cuda.Event` 绕开 profiler，两条路实测选出同一个 kernel，详见 [详细说明](1_rocm_torch_profiler_no_gpu_events/)。

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
