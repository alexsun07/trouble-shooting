# sglang 的 MXFP8 scale 改名把 NVFP4 expert block scale 静默丢掉 → 输出乱码

> **先读这一句**：「服务正常起来、没有报错、但输出是乱码」可以有很多原因（kernel 布局/flag 用错、
> gate/up 顺序反了、clamped-SwiGLU 当成 SiLU、scale 语义取反、aiter scale padding 未初始化……）。
> **出现这个现象不等于就是本条根因**，先用下面的「快速判断」确认参数确实没加载到，再套用本条。

## 现象

- sglang upstream（实测 `9f5d871251`，rocm/sgl-dev v0.5.21）跑 MiniMax-M3.1
  （`quant_method=mxfp8` + `moe_quant_algo=NVFP4`：MXFP8 linear + compressed-tensors NVFP4 routed experts）。
- 启动、加载、CUDA graph 全部正常，**日志里没有任何加载 warning**（没有 `Error loading weight`，
  也没有 `not found in params_dict`）。
- 输出乱码：chat 复读系统提示片段、`ai，ai，ai…`。换 attention kernel（Gluon / Triton）、
  KV dtype（FP8 / BF16）、MoE kernel（aiter FlyDSL / Triton）**全都一样**。
- 逐层 dump 对比正常版本：路由（topk id / weight）一致，但 MoE experts 输出不一致，后面某层 experts 输出全 0。

## 快速判断

1. 跟一个已知正常的版本（同权重）用同一 prompt 做逐层 dump（`--debug-tensor-dump-output-folder`，
   eager），用 [dump_compare.py](dump_compare.py) 找第一个偏离的模块。注意 dense / MoE 的输出在
   all-reduce 之前 dump，两边归约位置不同会差约 TP 倍，**这不是问题**——看下一层输入是否已偏离。
2. 偏离点在 MoE 时，在 `FusedMoE._weight_loader_impl` 里对某一层某个 expert 打印到达的
   `weight_name`：如果 `weight_packed` / `weight_global_scale` 到了，但 **`weight_scale` 从没到**，就是本条。
3. 再看参数值：未加载的参数是 `torch.empty`，**各 TP rank 上值一模一样**（真正加载的切片各 rank 不同）。

## 触发条件

- 充要：`DefaultModelLoader.load_weights_only` 里 `quant_config.get_name() == "mxfp8"`，且 checkpoint 里有
  **本来就叫 `.weight_scale`、却不是 MXFP8 scale** 的张量（这里是 compressed-tensors NVFP4 的 per-block scale）。
- 继承 `Fp8Config` 的 hybrid 配置（`get_name()` 仍返回 `mxfp8`）都会中招。
- 无关：attention backend、KV dtype、MoE kernel、TP 大小、GPU 平台（改名逻辑与平台无关，按机制推测
  NV 上也会中招；实测只在 ROCm MI355X）。
- sglang v0.5.17 没有这段改名逻辑，同样代码不触发。

## 根因

upstream loader 为兼容部分把 MXFP8 scale 存成 `weight_scale` 的 checkpoint，在 quant name 为 `mxfp8` 时把
**所有** `*.weight_scale` 改名为 `*.weight_scale_inv`。NVFP4 experts 的 block scale 因此被改成
`experts.w13_weight_scale_inv`，模型里没有这个参数；`MiniMaxM3SparseForConditionalGeneration._load_llm_weight`
的 expert 分支对找不到的参数**直接 return、不打 warning**，于是 block scale 保持未初始化。

## 修复

hybrid 配置跳过该改名（在 loader 里判断 MXFP8 linear + NVFP4 expert 的 hybrid 配置，例如一个 `_is_hybrid_mxfp8_nvfp4` 判定）。

无效的直觉做法（本次实际走过的弯路）：
- 换 Triton attention / BF16 KV / Triton MoE：kernel 不是问题，只是在浪费重启。
- 用 `/generate` 发纯文本看首 token：正常版本对 "The capital of France is" 的首 token 也是特殊 token
  `]<]minimax[>[`，这个测试没有判别力。
- 关掉 upstream 独有的融合（aiter fused qk-norm、fused qkv+index GEMM）：无关。
- 静态对比 config 解析（transformers 5.12 vs 5.19）：一致，无关。

## 影响范围

任何 `get_name()=="mxfp8"` 但混有其它格式、且其它格式 scale 恰好叫 `.weight_scale` 的 checkpoint。
更一般的教训：**loader 里「找不到参数就静默跳过」的分支会把加载问题伪装成数值问题**——遇到乱码时，
先确认每类参数都真的被加载过。
