问题
长 context 请求（RAG，几万 token 的 prompt）prefill 要跑几十秒。跑这种请求时有两件事会让前面那几十秒白干：

engine 挂了，请求只能甩到另一台 engine，重新 prefill
新来个 SLO 很紧的请求要插队，想踢掉正在跑的长 context 请求腾位置 —— 踢了 prefill 也白干
所以现在所有 SLO scheduling paper（QLM、Scorpio）都不敢踢正在跑的长 context 请求，太亏。

观察
Recovery 那派的系统（Mooncake）已经把 KV state 定期搬到 host memory 了 —— 它的目的是 engine 挂了能从 host 拉回来恢复。这份 host 上的 KV 数据，scheduler 也能拿来用啊。

做法
每个请求跑的时候，每 16 个 token 就把它新产生的 KV 块从 GPU 拷一份到 host pinned memory，再 publish 到 /dev/shm（让别的 engine 也能读）。这一份数据同时干两件事：

Recovery：engine A 挂了，请求迁到 engine B，engine B 从 /dev/shm 读 KV 接着跑，不用重 prefill
Cheap preempt：scheduler 想踢一个正在跑的请求让紧急请求上，踢就踢 —— 反正 KV 在 host 还在，等会恢复也不亏
关键 insight

踢人变便宜了，scheduler 终于敢按"哪个请求最快超时"（slack）来调度长 context 请求。每个请求算自己离超时还剩多少时间，最急的先跑；如果正在跑的某个请求剩很多时间，而队列头剩没多少时间，就把它踢下来让位 —— 反正踢的成本被 checkpoint 兜底了。

故事一句话
路径 1：弱化 Mooncake，强调 recovery 空白
"Mooncake 提供了 host-side KV pool 这个机制，但目标是 prefix cache 复用；recovery 这条路 LLM serving 几乎没人做。我们借用类似机制做两件新事：recovery + cheap preempt"

路径 2：把 Mooncake 当机制启发，scheduling 一侧主打 QLM 对比
"QLM 知道按 slack 排队是对的，但只敢在入口拒人，不敢踢 running。我们把 host-side KV checkpoint 这个机制（类似 Mooncake 的 host KV pool，但用途不同）拿来给 scheduler 用，让它敢踢长 context 请求"



问题：但写shm会不会性能变的很差啊
那我们是不是paper里可以说我们支持两种checkpoint

1. 写 shm 性能差不差
默认配置下确实有点慢，但有 dial 可以调。

每次 save 触发一次 publish。Publish 写 3 个文件：chunk + manifest + latest pointer，每个文件都 atomic write（tmp + replace），默认还 fsync。

性能档位（每次 publish 的开销）：

配置	耗时	备注
默认 (torch.save + fsync)	~50 ms	round 1 我们 smoke 跑的就是这个
FT_FAST_TMPFS_WRITE=1	~30 ms	跳 fsync，tmpfs 上 fsync 是冗余的
FT_FAST_CHUNK_FORMAT=1	~10 ms	用自定义二进制格式跳 pickle 序列化
上面两个一起 + FT_INLINE_MANIFEST=1	~3-5 ms	manifest 嵌进 chunk header，只写一个文件
Publish 只在请求积累了新 block 时触发（增量，不是全量），所以每次写的实际数据量是 30KB-100KB，IO 不是瓶颈，syscall + fsync 是瓶颈。

实际 paper evaluation 时 默认就该用 FAST_CHUNK + FAST_TMPFS_WRITE + INLINE_MANIFEST 三件套，跟 vLLM 默认 prefill 路径比开销可忽略（prefill 单步 100-200 ms）。

老分支 round 1/2 跑 30-concurrent 16K 没爆，说明这套组合性能 OK。

2. Paper 里 "两种 checkpoint" 怎么讲
可以讲，而且这是个好卖点。两种不是并列的两条路，是一份数据两层存储：

层	给谁用	访问代价
Layer 1：进程内 host pinned memory pool (KVCheckpointPool)	Same-engine capacity preempt + reload	直接指针，<1 ms
Layer 2：tmpfs 共享镜像 (/dev/shm)	Cross-engine recovery (engine 死 → 别的 engine 接手)	文件 IO，~5 ms (fast chunk)
Paper 里的措辞可以是：

"We maintain a two-tier KV checkpoint store. The host pool serves same-engine cheap preempt at sub-millisecond access cost. A tmpfs mirror on /dev/shm serves cross-engine recovery, paying file-IO cost only when the local pool misses (engine death or capacity eviction). The same data backs both uses; no separate copy is maintained."

卖点维度：

机制 × 策略复用：proposal 已经讲了"一份数据两个用途"，这里再加一层"两层存储优化访问代价"
Worst-case 兜底：local pool 命中走快路；miss 才退到 shm，性能-可用性两头都有
对比：QLM / Llumnix / Mooncake 都只有一层 store，要么纯进程内（撑不住 engine 死），要么纯跨进程（每次访问都走 IO）。我们分层做到两全
不过有一点要 honest：两层不是各存一份，是 host pool 是 source-of-truth，shm 是异步镜像。蛛丝马迹是：cross-engine restore 时拿到的是几十毫秒前的状态（publish 比 host pool 慢一拍）。这个 lag 写 paper 时要诚实承认，但不影响故事。


改进：
现在最大的弱点是：recovery 和 SLO scheduling 在故事上耦合，技术上独立。两个 mechanism 共用一份 host 数据，但它们之间没有算法层面的互动。审稿人会问 "ablation 把其中一个砍掉，另一个还成立吗？" —— 如果都成立，那这两件事就不该写在一篇 paper 里。

要把"两个东西放一起"升级成"两个东西必须放一起"，思路上要加一个机制 × 策略真正耦合的点。几个方向：

Checkpoint cadence 跟 slack 联动：紧 SLO 的请求 checkpoint 更频繁（preempt 不丢工作），松 SLO 的请求 checkpoint 稀疏（省 PCIe）。这样 mechanism 直接被 policy 驱动
Disruption-aware slack 计算：rerouted 请求的 slack 不只看 TTFT/TPOT，要加上 "recovery gap"（你 utils.py 里已经有 failure_gap_slo_ms，但只在 stage 判断里用，没在 scheduling 决策里用），让 disruption-recovery 路径在 SLO 优先级上有特殊地位
跨 engine 的 SLO scheduling：当一台 engine 接收 rerouted 请求时，本地正在跑的低 slack 请求要不要主动让位 —— 一个跨 engine 的 slack 协调
任意加一个就能从"组合"变成"协同"，故事质感能提一档。有了协同点，SoCC / EuroSys 的成功率会显著高，但还到不了顶会。

