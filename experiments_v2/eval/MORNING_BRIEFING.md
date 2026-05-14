# Morning briefing

## 后台任务（一觉醒来应该都完了）

- **bdbbyv7ad**：E_M3 完整 matrix，5 个 run × ~4min = ~20 min。**已完成**。
- **b53sakmd7**：SLO 校准 chain，4 个 calibration ≈ 50-60 min。状态见 `tail -30 /tmp/claude-1009/-home-yzhong76-code-my-vllm-serving-system/b6b6d10e-6333-4bb5-a610-a73e625bb2bc/tasks/b53sakmd7.output`

## E_M3 已完成数据（async fix 后 9 个 run）

| Baseline | TTFT P50 mean ± std | TTFT P95 mean ± std | TPOT P50 | Throughput |
|---|---|---|---|---|
| vllm_fcfs | 141.6 ± 19.3 | 416.8 ± 45.2 | 21.8 ± 0.0 | 47.9 ± 4.1 |
| reroute_no_ckpt | 141.2 ± 18.9 | 417.4 ± 42.9 | 21.7 ± 0.0 | 47.9 ± 4.1 |
| **ours** | **177.5 ± 36.5** | **458.8 ± 26.5** | **22.1 ± 0.1** | **47.9 ± 4.1** |

Paper 故事：**FT 机制开销可忽略**（throughput / TPOT 完全一样；TTFT P50 多 ~36ms ≈ 25%，但绝对值都在 100-200ms 量级，对长上下文 paper 故事无关紧要）。

跑 `python experiments_v2/eval/analysis/aggregate.py --experiment e_m3` 查最新表格。

## 待你拍板的事

1. **看 SLO 校准结果决定 S_TTFT / S_TPOT**：跑 `python experiments_v2/eval/analysis/aggregate.py --experiment slo_calib` 或者直接读 `experiments_v2/eval/results/slo_calib_*_metrics.json` 里的 `suggested_slo` 字段。
   - RULER 64K @ 0.04 是主校准；@ 0.02 是 cross-check
   - ShareGPT @ 0.15 / 0.1 同理
   - 两个 cross-check 数字接近就用主值；如果差 30%+ 用低 QPS 那个

2. **E_M1 QPS sweep 范围**：常规做法是从"远低于饱和"到"远高于饱和"扫 5-8 个点。需要服务时间 → 饱和点 QPS = 并发容量 / 服务时间。
   - 我估的 RULER 64K 单请求服务时间 15-25s，2 engine 并发容量 ~4-6 → 饱和点 ~0.2 req/s
   - 建议 sweep: 0.05, 0.1, 0.2, 0.3, 0.5, 1.0 req/s
   - ShareGPT 服务时间 ~5s，饱和点 ~1-2 req/s → sweep 0.1, 0.5, 1, 2, 4

3. **E_M1 每点 N**：N=50 给稳 P95 还行；N=100 更稳但跑久。3 seeds 配合可以聚合 N=150 → P95 可信

## 已就绪的脚本

```
experiments_v2/eval/
  scripts/
    e_m1_slo_sweep.py        # 主 SLO sweep。支持 4 baselines
    e_m3_overhead.py         # 已完整跑过
    e_d1_disruption_demo.py  # 已跑过
    slo_calibration.py       # 后台跑中
  workloads/
    workload_builder.py      # 数据集加载 + Poisson 到达 + 请求 schedule 拼装
  analysis/
    aggregate.py             # 聚合 JSON results 出 mean±std 表
    plot_em1.py              # 主图 SLO_met vs QPS 折线
```

## 跑 E_M1 主图的命令模板

```bash
# 单个 run
python experiments_v2/eval/scripts/e_m1_slo_sweep.py \
    --baseline ours \
    --dataset ruler_64k \
    --num-requests 50 \
    --arrival-rate-qps 0.2 \
    --seed 0 \
    --ttft-slo-ms <从校准取2x P95> \
    --tpot-slo-ms <从校准取2x P95>

# 矩阵驱动（你写个 shell loop 跑 baseline × qps × seed）
for baseline in vllm_fcfs reroute_no_ckpt ours; do
  for qps in 0.05 0.1 0.2 0.3 0.5 1.0; do
    for seed in 0 1 2; do
      python ... --baseline $baseline --arrival-rate-qps $qps --seed $seed ...
    done
  done
done
```

## 几个 paper-friendly 的事实

- `failover_gap_p95` 从 ours 跑数据：1.4-1.6s（E_D1 demo 已有）
- async fix 后 FT 开销：~36ms TTFT P50 (vs 之前 sync 模式的 97ms)
- 标准 baseline P95 在 ShareGPT 上 = 466ms 量级（5s 间隔下）
