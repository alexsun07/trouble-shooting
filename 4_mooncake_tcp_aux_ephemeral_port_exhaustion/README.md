# sglang PD（mooncake）长时间压测后 prefill 报 `Cannot assign requested address`，router 判 prefill 不健康

## 现象

单机 PD（mooncake 后端，机器没有 RDMA 网卡）在长 prompt 下持续压一段时间（实测 75k token prompt、并发 4~16，
跑了十几到二十分钟）之后：

- prefill 日志刷屏（一次压测 8.4 万行）：
  ```
  E... tcp_transport.cpp:863] TcpTransport::getConnection failed to create connection to <本机 IP>:<端口>. Error: connect: Cannot assign requested address
  ```
- 压测客户端报的是 router 的错，不提网络：
  ```
  ValueError: Warmup failed ... Service Unavailable ... "No available servers: No available prefill workers (all circuits open or unhealthy)"
  ```
- 宿主机 `ss -s` 里 TIME_WAIT 有几千到上万个，几乎全部连向 decode 的 4 个端口（每个 decode TP rank 一个，是 decode 的
  scheduler 进程在监听的 mooncake TCP 端口）。压测期间每 25 秒翻一倍（实测每个端口 1120 -> 2240 -> 4480）。

GSM8K 这类短时间、低并发的精度测试完全触发不到，所以 PD 可能"精度验证都通过、一压性能就挂"。

## 触发场景

三个条件同时满足：

1. mooncake 有一部分传输走它自己的 TCP 通道：PD 的 aux 元数据缓冲（输出 token、cached tokens、logprob、top-logprob、
   topk、hidden states、bootstrap room 等，`disaggregation/utils.py` 里 `MetadataBuffers` 默认 `device="cpu"`）在 host 内存。
   mooncake 的 HIP（或 NVLink）intra-node transport 只注册显存，所以发往 host 内存的写入只能走 TCP。KV、index-K、dense K/V、
   DSpark 草稿 KV、staging buffer 都在显存，走 HIP，不受影响。
2. mooncake 的 TCP transport 没开连接池（容器里自编译的 Mooncake `01d1eb2a`，`tcp_transport.h` 里 `enable_connection_pool_`
   默认 false）：每个传输 slice 新建一条 socket，传完马上 close。
3. 请求持续不断。实测 PP4 prefill -> TP4 decode，单个请求让 4 个端口合计新增约 150 个 TIME_WAIT（约 40 个/端口，
   与"4 个 PP stage x 每个 stage 约 10 块 aux"吻合，后者是数代码得出的推断）。TIME_WAIT 要 60 s 才释放；每个
   (本机 IP, 目标端口) 只有 `ip_local_port_range`（32768-60999，约 2.8 万）个源端口可用。

`tcp_tw_reuse=2`（Linux 默认）只对 loopback 生效，而 mooncake 连的是本机网卡 IP（不是 127.0.0.1），所以 TIME_WAIT 不能被复用。

注意它常和 PD staging 的 300 s 卡顿（`[STAGING] ... a scatter never arrived`）同时出现，那是另一个问题，不要混为一谈：
开连接池后端口报错消失，但 staging 卡顿照旧。

## 根因

mooncake TCP transport 关着连接池时一条连接只传一个 slice（`tcp_transport.cpp` 约 851-866 行新建、约 1081 行关闭），
sglang PD 每个请求、每个 PP stage、每个 decode rank 都要把 aux 写成若干个 slice。持续负载下新建连接的速率乘以
TIME_WAIT 的 60 s，超过单个目标端口可用的约 2.8 万个源端口，`connect()` 返回 `EADDRNOTAVAIL`。prefill 发不出 aux，
请求卡住，健康检查超时，router 打开熔断把 prefill 摘掉。

## 解决方案

prefill 和 decode 都设：

```bash
export MC_TCP_ENABLE_CONNECTION_POOL=1
```

mooncake 在 TCP transport 初始化时读这个变量（`tcp_transport.cpp` 约 618 行；值为 `0/false/no` 以外的任何值都打开），
连接空闲 60 s 才回收。实测（4 次独立压测）：75k prompt 并发到 32 跑完，`Cannot assign requested address` 为 0，
压测期间 TIME_WAIT 只有几十到一两百个。PD 启动脚本里 prefill 和 decode 都应默认带上它。

其他可选的办法（未实测）：

- `SGLANG_MOONCAKE_SEND_AUX_TCP=1`：aux 不走 mooncake，改走已有的 ZMQ 长连接。
- `SGLANG_HOST_IP=127.0.0.1`：让 mooncake 用 loopback 地址，于是 `tcp_tw_reuse=2` 生效。仅限单机 PD。

## 排查方法

- `tw_by_port.sh`（本目录）：按目标端口统计 TIME_WAIT。压测时每 25 s 跑一次，前几名端口数量持续翻倍就是这个问题。
- 确认是哪个进程在监听：`ss -tlnp | grep <端口>`，应是 decode 的 `sglang::schedul` 进程。
- 判断连接池是否起效要看持续压测时的 TIME_WAIT，不要看单个请求：实测开了连接池后，服务刚起来时的第一个 75k 请求照样留下
  约 140 个 TIME_WAIT（与不开时相当，原因未查明），但随后几十分钟的压测里总数一直在 15~100 之间；不开时同阶段是几千且持续翻倍。

## 走过的弯路

- 一开始只看到 router 的 "No available prefill workers (all circuits open or unhealthy)"，像是 prefill 崩了；prefill 进程其实还活着，
  要去 prefill 日志里找 `tcp_transport.cpp`。
- 用 `/proc/<scheduler pid>/environ` 检查 `MC_TCP_ENABLE_CONNECTION_POOL` 是否传进 scheduler，结果看不到这个变量（连
  `HF_HUB_CACHE` 等也看不到；它的父进程 launch_server 能看到）。为什么看不到没有查明，但变量实际是生效的（TIME_WAIT
  不再累积、端口报错为 0），所以 `/proc/<pid>/environ` 不能作为判断依据，要看 TIME_WAIT 是否还在累积。
- 加大 staging 环（`SGLANG_DISAGG_STAGING_POOL_SIZE_MB`）对这个问题无效，它针对的是 staging 卡顿那个问题；8 GB 时 decode 直接 OOM
  （staging 环在 KV cache 分配之后申请）。
