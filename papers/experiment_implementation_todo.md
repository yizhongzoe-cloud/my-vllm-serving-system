# 实验框架升级 — 完整 TODO List (v2)

> 从当前 experiments/ 升级到 experiment_design.md 方案的全部改动项
> 新代码写到 `experiments_v2/`，不动现有 `experiments/`

---

## 概览：文件清单

```
experiments_v2/
├── config_1b.yaml                  # [新建] 1B 模型配置（快速开发测试）
├── config_8b.yaml                  # [新建] 8B 模型配置（主力实验）
├── config_70b.yaml                 # [新建] 70B 模型配置（规模验证）
├── datasets/                       # [新建] 数据集管理
│   ├── download.py                 #   下载 + 预处理脚本
│   ├── loader.py                   #   统一加载器（ShareGPT/CNN-DM/Alpaca）
│   └── cached/                     #   预处理后的缓存文件
├── workloads.py                    # [重写] 基于真实数据集的 trace 生成
├── calibrate.py                    # [新建] SLO 校准 + 负载饱和点探测
├── run.py                          # [改] per-request SLO、强制输出长度、新 config
├── suite.py                        # [改] 多模型、百分比负载、SLO scale 维度
├── analyze.py                      # [大改] 新增 CDF/热力图/散点图/时间线 + 多模型合并
├── profile_checkpoint_costs.py     # [改] 自动获取模型 KV 几何参数
└── run_all.sh                      # [新建] 一键执行脚本
```

---

## 三个模型配置的定位

| 配置 | 模型 | TP | dp_size | GPU | 用途 |
|---|---|---|---|---|---|
| config_1b.yaml | Llama-3.2-1B-Instruct | 1 | 2 | 2×A100 | **开发测试**：验证全流程、调试 pipeline、快速迭代。不写入论文。 |
| config_8b.yaml | Llama-3.1-8B-Instruct | 1 | 4 | 4×A100 | **主力实验**：E1a 全扫、E2-E6 8B 部分。论文主要结果。 |
| config_70b.yaml | Llama-3.1-70B-Instruct | 4 | 2 | 8×A100 | **规模验证**：E1b 精简矩阵、E2/E4/E5 70B 部分。论文验证结果。 |

开发流程：先用 1B 跑通整个 pipeline → 切到 8B 跑正式实验 → 70B 做验证。

---

## Phase 0: 数据集准备

### T0.1 创建 `experiments_v2/datasets/download.py`

下载并预处理三个数据集，输出统一格式的 JSONL 文件。

```python
# 每条记录的格式:
{
    "id": "sharegpt-00001",
    "dataset": "sharegpt",
    "prompt": "实际 prompt 文本",
    "prompt_tokens": 235,           # Llama tokenizer 计数
    "expected_output_tokens": 180,  # 真实 output token 数
}
```

**Tokenizer 统一**：所有数据集的 token 计数必须用 `meta-llama/Llama-3.1-8B-Instruct` 的 tokenizer（Llama-3.1 和 3.2 共享 tokenizer）。在 download.py 顶部加载一次，全局复用。

**具体任务**：

- [ ] **T0.1.1** ShareGPT 下载 + 处理
  - 来源：`anon8231489123/ShareGPT_Vicuna_unfiltered` (HuggingFace)
  - **多轮对话处理**：拼接完整对话历史到最后一个 user turn，作为 prompt；最后一个 assistant turn 的 token 数作为 expected_output_tokens
    ```python
    # 示例：对话有 [user1, assistant1, user2, assistant2, user3, assistant3]
    # prompt = "user1\nassistant1\nuser2\nassistant2\nuser3"
    # expected_output_tokens = len(tokenize(assistant3))
    ```
  - 只取对话轮次 >= 2 的样本（确保 prompt 有足够上下文）
  - 过滤：丢弃 prompt_tokens + expected_output_tokens > 4096 的样本
  - 采样 **5000** 条（预留足够量给高 RPS 场景）
  - 保存为 `cached/sharegpt_5000.jsonl`

- [ ] **T0.1.2** CNN/DailyMail 下载 + 处理
  - 来源：`cnn_dailymail` (HuggingFace datasets, version 3.0.0)
  - 处理：article + "\n\nSummarize the above article in one paragraph." 作为 prompt，highlights 的 token 数作为 expected_output_tokens
  - 过滤：丢弃 prompt_tokens + expected_output_tokens > 4096 的样本
  - 采样 3000 条，保存为 `cached/cnndm_3000.jsonl`

- [ ] **T0.1.3** Alpaca 下载 + 处理
  - 来源：`tatsu-lab/alpaca` (HuggingFace)
  - 处理：如果有 input 字段，拼接为 "instruction\n\ninput"；否则只用 instruction。output 的 token 数作为 expected_output_tokens
  - 不做长度过滤（Alpaca 本身较短）
  - 全部 52k 条保存为 `cached/alpaca_full.jsonl`

- [ ] **T0.1.4** 验证脚本
  - 打印每个数据集的 token 长度统计：mean, std, P50, P95, P99, min, max（input 和 output 分开）
  - 画直方图确认分布形状（应该是重尾）
  - 检查 expected_output_tokens 的分布是否合理（ShareGPT 应该 mean ~200+，CNN/DM ~100，Alpaca ~100-300）

- [ ] **T0.1.5**（可选）Azure LLM Trace 下载
  - 来源：Azure LLM inference traces（Splitwise/Vidur 公开数据）
  - 处理：提取请求到达时间戳序列
  - 保存为 `cached/azure_coding_trace.json` 和 `cached/azure_chatting_trace.json`
  - 标记为 nice-to-have，不阻塞主流程

### T0.2 创建 `experiments_v2/datasets/loader.py`

统一加载器，所有数据集用同一个接口。

- [ ] **T0.2.1** 接口设计
  ```python
  def load_dataset(
      dataset_name: str,       # "sharegpt" | "cnndm" | "alpaca"
      dataset_path: str,       # JSONL 文件路径
      max_samples: int = None, # 最多加载多少条
      seed: int = 42,          # 可复现采样
  ) -> list[dict]:
      """返回 [{id, dataset, prompt, prompt_tokens, expected_output_tokens}, ...]"""
  ```

- [ ] **T0.2.2** 循环采样支持：当请求数 > 数据集大小时，自动 wrap around（有放回循环）
  ```python
  def sample_from_dataset(
      dataset: list[dict],
      n: int,
      rng: random.Random,
  ) -> list[dict]:
      """如果 n > len(dataset)，循环采样"""
      return [dataset[rng.randint(0, len(dataset)-1)] for _ in range(n)]
  ```

---

## Phase 1: 配置文件

### T1.1 创建 `config_1b.yaml`（开发测试用）

- [ ] **T1.1.1** 模型段
  ```yaml
  model: "meta-llama/Llama-3.2-1B-Instruct"
  max_model_len: 2048      # 1B 用短一些，加快速度
  gpu_memory_utilization: 0.45
  dtype: "float16"
  enforce_eager: true
  dp_size: 2
  ```

- [ ] **T1.1.2** 工作负载段（和 8B/70B 共享同一套 workload 定义，只是 max_model_len 不同）
  ```yaml
  workloads:
    W1_Chat:
      dataset: "sharegpt"
      dataset_path: "experiments_v2/datasets/cached/sharegpt_5000.jsonl"
      tpot_slo_ms: 100.0
    W2_Summary:
      dataset: "cnndm"
      dataset_path: "experiments_v2/datasets/cached/cnndm_3000.jsonl"
      tpot_slo_ms: 200.0
    W3_Instruct:
      dataset: "alpaca"
      dataset_path: "experiments_v2/datasets/cached/alpaca_full.jsonl"
      tpot_slo_ms: 50.0
    W4_Mixed:
      mix:
        - {workload: "W1_Chat", weight: 0.50}
        - {workload: "W3_Instruct", weight: 0.20}
        - {workload: "W2_Summary", weight: 0.30}
  ```

- [ ] **T1.1.3** 负载段
  ```yaml
  load_levels:
    Light:    {pct: 0.25, rps: null}  # calibrate 后自动填入
    Moderate: {pct: 0.40, rps: null}
    Heavy:    {pct: 0.55, rps: null}
  ```

- [ ] **T1.1.4** SLO 段
  ```yaml
  slo:
    ttft_ms: null            # = 5 × TTFT_base，calibrate 后填入
    failure_gap_ms: null     # = 3 × max(Gap_base across workloads)
    # TPOT 在 workload 级别已定义
  ```

- [ ] **T1.1.5** Baseline 段（6 个）
  ```yaml
  baselines:
    No-FT:
      scheduling_policy: "fcfs"
      enable_checkpointing: false
      fixed_checkpoint_blocks: 0
    Periodic-Low:
      scheduling_policy: "fault_tolerant"
      enable_checkpointing: true
      fixed_checkpoint_blocks: 10
    Periodic-High:
      scheduling_policy: "fault_tolerant"
      enable_checkpointing: true
      fixed_checkpoint_blocks: 1
    Benders-Only:
      scheduling_policy: "ft_benders_centralized"
      enable_checkpointing: true
      fixed_checkpoint_blocks: 10
    Adaptive-Only:
      scheduling_policy: "fault_tolerant"
      enable_checkpointing: true
      fixed_checkpoint_blocks: 0
    Our-System:
      scheduling_policy: "ft_benders_centralized"
      enable_checkpointing: true
      fixed_checkpoint_blocks: 0
  ```

- [ ] **T1.1.6** 故障注入段
  ```yaml
  fault_timing:
    none: null
    F1_Early: 60.0
    F2_Mid: 150.0
    F3_Late: 240.0
  ```

- [ ] **T1.1.7** 运行参数段
  ```yaml
  run_duration_sec: 300.0
  warmup_sec: 30.0
  seeds: [42, 123, 456]
  request_timeout_sec: 120.0
  checkpoint_pool_bytes: 8589934592  # 8 GB（1B 模型够用）
  force_output_len: true             # 强制生成 expected_output_tokens 个 token
  ```

- [ ] **T1.1.8** 实验定义段
  ```yaml
  experiments:
    E0_Smoke:
      description: "Pipeline 验证"
      baselines: ["No-FT", "Our-System"]
      workloads: ["W1_Chat"]
      load_levels: ["Moderate"]
      faults: ["none", "F2_Mid"]
      # 不含 slo_scales → 用默认 SLO

    E1a_Main:
      description: "端到端 (主力模型)"
      baselines: ["No-FT", "Periodic-Low", "Periodic-High",
                   "Benders-Only", "Adaptive-Only", "Our-System"]
      workloads: ["W1_Chat", "W2_Summary", "W3_Instruct", "W4_Mixed"]
      load_levels: ["Light", "Moderate", "Heavy"]
      faults: ["none", "F1_Early", "F2_Mid", "F3_Late"]

    E2_Recovery:
      description: "恢复时间分解"
      baselines: ["No-FT", "Periodic-Low", "Periodic-High", "Our-System"]
      workloads: ["W1_Chat", "W2_Summary"]
      load_levels: ["Moderate"]
      faults: ["F2_Mid"]

    E3_Ablation:
      description: "消融实验"
      baselines: ["Periodic-Low", "Periodic-High", "Benders-Only",
                   "Adaptive-Only", "Our-System"]
      workloads: ["W1_Chat", "W4_Mixed"]
      load_levels: ["Moderate", "Heavy"]
      faults: ["none", "F2_Mid"]

    E4_Checkpoint_Tradeoff:
      description: "Checkpoint 开销 vs 恢复收益"
      baselines: ["No-FT", "Periodic-Low", "Periodic-High", "Our-System"]
      workloads: ["W1_Chat"]
      load_levels: ["Moderate", "Heavy"]
      faults: ["none", "F2_Mid"]

    E5_Controller:
      description: "Solver 开销"
      baselines: ["Our-System"]
      workloads: ["W1_Chat", "W2_Summary", "W3_Instruct", "W4_Mixed"]
      load_levels: ["Light", "Moderate", "Heavy"]
      faults: ["none"]

    E6_SLO_Sensitivity:
      description: "SLO 敏感度"
      baselines: ["No-FT", "Our-System"]
      workloads: ["W1_Chat", "W4_Mixed"]
      load_levels: ["Moderate"]
      faults: ["F2_Mid"]
      slo_scales:
        Tight:    {ttft_mult: 3.0,  gap_mult: 1.5}
        Moderate: {ttft_mult: 5.0,  gap_mult: 3.0}
        Loose:    {ttft_mult: 10.0, gap_mult: 5.0}
  ```

### T1.2 创建 `config_8b.yaml`

- [ ] **T1.2.1** 从 config_1b.yaml 复制，修改：
  - model → `meta-llama/Llama-3.1-8B-Instruct`
  - max_model_len → 4096
  - gpu_memory_utilization → 0.90
  - dp_size → 4
  - checkpoint_pool_bytes → 34359738368 (32 GB)

### T1.3 创建 `config_70b.yaml`

- [ ] **T1.3.1** 从 config_8b.yaml 复制，修改：
  - model → `meta-llama/Llama-3.1-70B-Instruct`
  - dp_size → 2
  - checkpoint_pool_bytes → 137438953472 (128 GB)
  - experiments 只保留 E1b_Main（精简矩阵）+ E2/E4/E5 中的 70B 部分

- [ ] **T1.3.2** E1b_Main 定义
  ```yaml
  E1b_Main:
    description: "端到端 (70B 验证)"
    baselines: ["No-FT", "Periodic-High", "Our-System"]
    workloads: ["W1_Chat", "W4_Mixed"]
    load_levels: ["Moderate", "Heavy"]
    faults: ["none", "F2_Mid"]
  ```

---

## Phase 2: workloads.py 重写

### T2.1 核心数据结构

- [ ] **T2.1.1** `RequestSpec` 保持原有字段，新增：
  ```python
  dataset: str = ""        # 来源数据集名 ("sharegpt", "cnndm", "alpaca")
  original_id: str = ""    # 数据集中的原始 ID
  ```

### T2.2 数据集驱动的 trace 生成

- [ ] **T2.2.1** `generate_trace()` 签名改造
  ```python
  def generate_trace(
      workload_config: dict,      # 含 dataset, dataset_path, tpot_slo_ms
      rps: float,
      duration_sec: float,
      seed: int,
      slo_config: dict,           # 全局 SLO (ttft_ms, failure_gap_ms)
      warmup_sec: float = 0.0,
      all_workload_configs: dict = None,  # 仅 Mixed 时需要，传全部 workloads
  ) -> list[RequestSpec]:
  ```

- [ ] **T2.2.2** 单数据集 trace 生成
  - 从 JSONL 加载数据集（通过 loader.py）
  - 生成到达时间（Poisson，复用 `_poisson_arrivals()`）
  - 每个到达时刻：从数据集有放回采样一条，构造 RequestSpec
    - prompt_text = 真实 prompt
    - prompt_len = 真实 prompt_tokens
    - expected_output_len = 真实 expected_output_tokens
    - tpot_slo_ms = workload_config["tpot_slo_ms"]
    - ttft_slo_ms = slo_config["ttft_ms"]
    - failure_gap_slo_ms = slo_config["failure_gap_ms"]
  - 当请求数 > 数据集大小时，自动循环采样

- [ ] **T2.2.3** Mixed workload trace 生成
  ```python
  def _generate_mixed_trace(...):
      # 检测 workload_config 是否有 "mix" 键
      # 如果有，每次到达时按 weight 采样 workload 类型
      # 从对应 workload 的 dataset 采样一条
      # tpot_slo_ms 用该 workload 自己的值（per-request 差异化）
  ```

- [ ] **T2.2.4** 删除 `_make_prompt()`、`_PROMPT_TEMPLATES`、`_TOPICS`

### T2.3 保留的部分

- [ ] `_poisson_arrivals()` — 不改
- [ ] `_bursty_arrivals()` — 不改（W4_Mixed 目前用 Poisson，bursty 留作备用）

---

## Phase 3: calibrate.py（新建）

**执行顺序（内部有依赖，必须串行）**：
```
T3.1 find_saturation_rps() → 拿到 RPS_sat
T3.1.2 compute_load_levels() → 算出 Light/Moderate/Heavy 的 RPS
T3.2.1 measure_baseline_latency() → 拿到 TTFT_base
T3.2.2 measure_recovery_gap_base() → 用 Moderate RPS 跑，拿到 Gap_base
T3.3.1 calibrate_config() → 回写 config
```

### T3.1 RPS 饱和点探测

- [ ] **T3.1.1** `find_saturation_rps(config, workload, port) -> float`
  - 用 No-FT baseline，不注入故障
  - 测试 RPS 列表：[0.5, 1, 2, 4, 6, 8, 10, 15, 20, 30]（可提前终止）
  - 每个 RPS 跑 **60s**（短时间足以判断是否饱和，不需要 300s）
  - 饱和条件：TPOT P95 > 2× TPOT_base 或 SLO violation > 10%
  - 返回饱和前的最大 RPS 作为 RPS_sat

- [ ] **T3.1.2** `compute_load_levels(rps_sat, dp_size, pct_config) -> dict`
  ```python
  # 输入: rps_sat=40, dp_size=4, pct_config={Light: 0.25, Moderate: 0.40, Heavy: 0.55}
  # 输出: {Light: 10.0, Moderate: 16.0, Heavy: 22.0}
  # 公式: rps = pct × rps_sat
  ```

### T3.2 SLO 基准测量

- [ ] **T3.2.1** `measure_baseline_latency(config, workload, port) -> dict`
  - 用 No-FT baseline，RPS = 0.1（近乎单请求）
  - 跑 30s，取 TTFT 和 TPOT 的 **P50** 作为 base
  - 返回 `{ttft_base_ms, tpot_base_ms}`

- [ ] **T3.2.2** `measure_recovery_gap_base(config, workload, port) -> float`
  - **前置条件**：T3.1 已完成，Moderate RPS 已知
  - 用 Periodic-High baseline + F2_Mid 故障 + Moderate 负载
  - 跑一次完整实验（300s），解析 recoveries.json
  - 取恢复请求的 failover gap **P50** 作为 Gap_base
  - 返回 Gap_base_ms

### T3.3 自动填入配置

- [ ] **T3.3.1** `calibrate_config(config_path, port) -> None`
  - 对每个**非 Mixed** workload (W1, W2, W3) 执行 T3.1 + T3.2
  - W4_Mixed 的 RPS_sat = min(W1, W2, W3 各自 RPS_sat 按 weight 加权)
  - TTFT = 5 × **max**(各 workload 的 TTFT_base)
  - Gap SLO = 3 × **max**(各 workload 的 Gap_base) ← **取 max，保守策略**
  - 将 rps 和 SLO 值回写到 config YAML
  - 输出校准报告（表格：每个 workload 的 RPS_sat, TTFT_base, TPOT_base, Gap_base）

### T3.4 命令行接口

- [ ] **T3.4.1** CLI
  ```bash
  python experiments_v2/calibrate.py \
      --config experiments_v2/config_1b.yaml \
      --port 8300 \
      --output experiments_v2/config_1b_calibrated.yaml
  ```
  支持 `--skip-recovery`（跳过 recovery prescan，加速开发测试时的校准）

---

## Phase 4: run.py 改动

### T4.1 强制输出长度（修复 P1）

- [ ] **T4.1.1** `_send_single_request()` 的 payload 新增：
  ```python
  if config.get("force_output_len", False):
      payload["min_tokens"] = spec.expected_output_len
  # max_tokens 已有 = spec.expected_output_len
  ```
  这确保模型不会提前 EOS，实际生成 token 数 = expected_output_len。

- [ ] **T4.1.2** 如果 vLLM 不支持 `min_tokens`，备选方案：`"ignore_eos": True`

### T4.2 per-request SLO 确认

- [ ] **T4.2.1** 确认 `_send_single_request()` payload 中 `tpot_slo_ms` 从 `spec.tpot_slo_ms` 读取（已有逻辑，确认 Mixed workload 下不同请求有不同值）

- [ ] **T4.2.2** 确认 `RequestResult.finalize_metrics()` 用 per-request `tpot_slo_ms` 判断（已有逻辑）

### T4.3 配置结构适配

- [ ] **T4.3.1** `_build_server_cmd()` 适配新 config 结构
  - `--default-tpot-slo-ms` 不再传全局值，改为传 workload 级别最严的（min across workloads）作为 server 端默认值
  - 实际 per-request SLO 通过 payload 传
  - `--checkpoint-pool-bytes` 从 config 读（不同模型不同值）
  - `--ft-checkpoint-cost-profile` 指向对应模型的 profile JSON

- [ ] **T4.3.2** `main()` 适配
  - load_levels 现在是 `{pct, rps}` 字典，读 `rps` 字段获取实际 RPS
  - `generate_trace()` 调用传入新参数（workload_config 含 dataset_path）
  - 检测 Mixed workload（有 `mix` 键）时调用 mixed trace 生成

### T4.4 新增输出字段

- [ ] **T4.4.1** `RequestResult` 新增 `dataset: str = ""`
- [ ] **T4.4.2** `_CSV_FIELDS` 列表新增 `"dataset"`
- [ ] **T4.4.3** `to_csv_row()` 输出 dataset 字段

### T4.5 E6 SLO scale 支持

- [ ] **T4.5.1** `main()` 新增 `--slo-scale` CLI 参数（可选）
  - 如果指定（如 `--slo-scale Tight`），从 config 的 `slo_scales.Tight` 读取倍数
  - 将 TTFT SLO 和 Gap SLO 按倍数缩放后覆盖默认值
  - 不指定则用默认 SLO

### T4.6 其他

- [ ] **T4.6.1** request_timeout_sec 默认值从 60 改为 120
- [ ] **T4.6.2** warmup_sec 默认值从 5 改为 30

---

## Phase 5: suite.py 改动

### T5.1 多模型支持

- [ ] **T5.1.1** 不在 suite 内做多模型循环，每个模型独立调用 suite.py
- [ ] **T5.1.2** 结果目录结构加模型层级：`results_v2/{model_tag}/{exp_name}/{baseline}/...`
  - model_tag 从 config 自动提取（如 "1B", "8B", "70B"）
  - 便于 analyze.py 合并多模型结果

### T5.2 负载级别

- [ ] **T5.2.1** `build_run_matrix()` 中 load_levels 读取 `rps` 值
  - 如果 rps 为 null，报错并提示先跑 calibrate.py
- [ ] **T5.2.2** 传给 run.py 的 `--load` 参数传 load level name，run.py 内部查表

### T5.3 SLO scale 维度（E6 支持）

- [ ] **T5.3.1** 如果实验定义含 `slo_scales`，将其加入笛卡尔积
  ```python
  # E6: baselines × workloads × loads × faults × slo_scales × seeds
  # run.py 额外传 --slo-scale Tight
  ```
- [ ] **T5.3.2** 输出目录加 slo_scale 层级：`.../Moderate/F2_Mid/Tight/42/`

### T5.4 校准集成

- [ ] **T5.4.1** `--calibrate` flag：先调 calibrate.py，再跑实验

### T5.5 其他

- [ ] **T5.5.1** `--resume` 逻辑不变
- [ ] **T5.5.2** 未校准检测：如果 config 中任何 load_level 的 rps 为 null，打印错误退出

---

## Phase 6: analyze.py 改动

### T6.1 多模型结果合并

- [ ] **T6.1.1** `load_all_runs()` 支持递归搜索多模型子目录
  - 输入 `results_v2/` → 自动发现 `results_v2/1B/E2_Recovery/...` 和 `results_v2/8B/E2_Recovery/...`
  - 在 run metadata 中添加 `model_tag` 字段（从目录结构或 run_meta.json 的 model 字段提取）

- [ ] **T6.1.2** `--results-dir` 支持多个路径
  ```bash
  python analyze.py results_v2/8B/E2_Recovery results_v2/70B/E2_Recovery --output figures_v2/E2
  ```

### T6.2 现有图表适配

- [ ] **T6.2.1** `BASELINE_STYLE` 更新（名称改了）
  ```python
  BASELINE_STYLE = {
      "No-FT":          {"color": "#888888", "marker": "x", "linestyle": "--"},
      "Periodic-Low":   {"color": "#E69F00", "marker": "s", "linestyle": "-"},
      "Periodic-High":  {"color": "#D55E00", "marker": "^", "linestyle": "-"},
      "Benders-Only":   {"color": "#009E73", "marker": "D", "linestyle": "-"},
      "Adaptive-Only":  {"color": "#CC79A7", "marker": "p", "linestyle": "-"},
      "Our-System":     {"color": "#0072B2", "marker": "o", "linestyle": "-"},
  }
  ```

- [ ] **T6.2.2** `plot_goodput_by_load()` X 轴从 `["Low", "Medium", "High"]` 改为 `["Light", "Moderate", "Heavy"]`
- [ ] **T6.2.3** `plot_ablation()` baseline 列表更新
- [ ] **T6.2.4** `plot_checkpoint_tradeoff()` baseline 列表更新

### T6.3 新增：Figure 6 — Recovery Gap CDF

- [ ] **T6.3.1** `plot_recovery_gap_cdf(runs, output_dir)`
  - 从 requests.csv 提取 direct-hit 请求的 max_gap_ms
  - 每个 baseline 一条 CDF 线
  - X 轴 log scale，Y 轴 0-1
  - 垂直虚线标注 Gap SLO 阈值
  - 如果有多模型数据，按模型分 subplot

### T6.4 新增：Figure 8 — 消融热力图

- [ ] **T6.4.1** `plot_ablation_heatmap(runs, output_dir)`
  - 行 = baselines (5)
  - 列 = (workload × load × fault) 组合
  - cell = SLO violation rate (%)
  - 颜色映射：绿(低) → 红(高)

### T6.5 新增：Figure 9 — Tradeoff 散点图

- [ ] **T6.5.1** `plot_checkpoint_tradeoff_scatter(runs, output_dir)`
  - X 轴：无故障 goodput 相对 No-FT 的下降百分比（开销）
  - Y 轴：有故障 goodput 相对 No-FT 的提升百分比（恢复收益）
  - 每个 baseline 一个带标签的点
  - **配对逻辑**：按 (baseline, workload, load) group，找 fault=none 和 fault=F2_Mid 的 pair
  - 如果有多模型，按模型分 subplot

### T6.6 新增：Figure 10 — Goodput 时间线

- [ ] **T6.6.1** `plot_goodput_timeline(runs, output_dir)`
  - 从 requests.csv 的 (end_time, output_tokens) 计算滑动窗口 goodput
  - 窗口大小 5s，步长 1s
  - X 轴：时间 (s)，红色垂直虚线标注故障时刻
  - Y 轴：goodput (tok/s)
  - 数据线：No-FT, Periodic-High, Ours（3 条）
  - 选一个代表性 seed（第一个）

### T6.7 新增：Figure 13/14 — SLO 敏感度

- [ ] **T6.7.1** `plot_slo_sensitivity(runs, output_dir)`
  - X 轴：SLO 松紧级别 (Tight / Moderate / Loose)
  - Y 轴双轴：goodput (left) + SLO violation rate (right)
  - 分组：No-FT vs Ours
  - **数据来源**：E6 实验（含 slo_scale 元数据）

- [ ] **T6.7.2** `plot_gap_slo_sensitivity(runs, output_dir)`
  - X 轴：Gap SLO 倍数 (1× ~ 5× Gap_base)
  - Y 轴：恢复请求中满足 Gap SLO 的比例
  - 每个 baseline 一条线
  - **不需要额外 runs**：从 E2 数据的 max_gap_ms 在不同阈值下重新计算

### T6.8 图表注册

- [ ] **T6.8.1** 更新 `EXPERIMENT_PLOTS`
  ```python
  EXPERIMENT_PLOTS = {
      "E0_Smoke":               ["goodput_by_load", "slo_violation"],
      "E1a_Main":               ["goodput_by_load", "slo_violation", "failover_gap"],
      "E1b_Main":               ["goodput_by_load"],
      "E2_Recovery":            ["recovery_breakdown", "recovery_gap_cdf"],
      "E3_Ablation":            ["ablation", "ablation_heatmap"],
      "E4_Checkpoint_Tradeoff": ["checkpoint_tradeoff_scatter", "goodput_timeline"],
      "E5_Controller":          ["controller_overhead"],
      "E6_SLO_Sensitivity":     ["slo_sensitivity", "gap_slo_sensitivity"],
  }
  ```

- [ ] **T6.8.2** 更新 `plot_dispatch` 字典注册所有新函数

---

## Phase 7: Profiling

### T7.1 profile 脚本改动

- [ ] **T7.1.1** `KVCheckpointBenchmark` 自动获取模型 KV 几何参数
  ```python
  from transformers import AutoConfig
  cfg = AutoConfig.from_pretrained(model_name)
  num_kv_heads = cfg.num_key_value_heads    # GQA heads
  head_dim = cfg.hidden_size // cfg.num_attention_heads
  num_layers = cfg.num_hidden_layers
  ```
  不再写死 Llama-3.2-1B 的参数（num_kv_heads=8, head_size=64, num_layers=32）。

### T7.2 执行 Profiling

- [ ] **T7.2.1** 1B: `profile_checkpoint_costs.py --model meta-llama/Llama-3.2-1B-Instruct --output experiments_v2/checkpoint_cost_profile_1b.json`
- [ ] **T7.2.2** 8B: `profile_checkpoint_costs.py --model meta-llama/Llama-3.1-8B-Instruct --output experiments_v2/checkpoint_cost_profile_8b.json`
- [ ] **T7.2.3** 70B: `profile_checkpoint_costs.py --model meta-llama/Llama-3.1-70B-Instruct --output experiments_v2/checkpoint_cost_profile_70b.json`
- [ ] **T7.2.4** 对每个 profile 跑 `inspect_checkpoint_profile.py` 验证单调性/凸性

---

## Phase 8: 执行脚本

### T8.1 创建 `run_all.sh`

- [ ] **T8.1.1** 完整执行流程 + 错误处理
  ```bash
  #!/bin/bash
  set -euo pipefail  # 任何错误立即停止

  PYTHON=python
  PORT=8300

  # ======= Phase 0: 数据集 =======
  echo "=== Downloading datasets ==="
  $PYTHON experiments_v2/datasets/download.py

  # ======= Phase 0: Profiling =======
  echo "=== Profiling 1B ==="
  $PYTHON experiments_v2/profile_checkpoint_costs.py \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --output experiments_v2/checkpoint_cost_profile_1b.json --port $PORT

  # (8B 和 70B profiling 类似，按需取消注释)

  # ======= Phase 0: 校准 =======
  echo "=== Calibrating 1B ==="
  $PYTHON experiments_v2/calibrate.py \
      --config experiments_v2/config_1b.yaml \
      --port $PORT \
      --output experiments_v2/config_1b_calibrated.yaml

  # ======= Phase 1: Smoke Test =======
  echo "=== Smoke Test ==="
  $PYTHON experiments_v2/suite.py \
      --config experiments_v2/config_1b_calibrated.yaml \
      --experiment E0_Smoke --port $PORT

  echo "=== Smoke Test Analysis ==="
  $PYTHON experiments_v2/analyze.py \
      results_v2/1B/E0_Smoke --output figures_v2/1B/E0_Smoke

  echo "Smoke test done. Check figures_v2/1B/E0_Smoke/ before continuing."
  echo "Press Enter to continue to full experiments, or Ctrl+C to abort."
  read

  # ======= Phase 2: 正式实验 (8B) =======
  echo "=== Calibrating 8B ==="
  $PYTHON experiments_v2/calibrate.py \
      --config experiments_v2/config_8b.yaml \
      --port $PORT \
      --output experiments_v2/config_8b_calibrated.yaml

  for EXP in E1a_Main E3_Ablation E2_Recovery E4_Checkpoint_Tradeoff E5_Controller E6_SLO_Sensitivity; do
      echo "=== Running $EXP (8B) ==="
      $PYTHON experiments_v2/suite.py \
          --config experiments_v2/config_8b_calibrated.yaml \
          --experiment $EXP --resume --port $PORT

      echo "=== Analyzing $EXP ==="
      $PYTHON experiments_v2/analyze.py \
          results_v2/8B/$EXP --output figures_v2/8B/$EXP
  done

  # ======= Phase 3: 70B 验证 =======
  # (类似，按需执行)

  echo "=== All done ==="
  ```

---

## Phase 9: 验证

### T9.1 单元测试

- [ ] **T9.1.1** 数据集加载器：每个数据集加载后 token 统计合理
  - ShareGPT: prompt mean > 100, output mean > 100
  - CNN/DM: prompt mean > 500, output mean > 50
  - Alpaca: prompt mean > 20, output mean > 20

- [ ] **T9.1.2** workloads.py：
  - `generate_trace()` 返回正确格式的 RequestSpec 列表
  - Mixed trace 的 dataset 字段分布与 weight 一致（容差 5%）
  - 循环采样：请求数 > 数据集大小时不报错

- [ ] **T9.1.3** calibrate.py：对 mock 数据能正确计算 RPS_sat 和 SLO 值

### T9.2 集成测试（用 1B 模型快速验证）

- [ ] **T9.2.1** 端到端 Smoke：config_1b + No-FT + W1_Chat + Moderate + F2_Mid
  - 验证：server 启动、ShareGPT 真实 prompt 发送、故障注入、恢复、metrics.json
  - 验证：output_tokens ≈ expected_output_len（force_output_len 生效）

- [ ] **T9.2.2** Mixed workload：config_1b + Our-System + W4_Mixed + Moderate + none
  - 验证：requests.csv 中不同请求有不同 tpot_slo_ms (50/100/200)
  - 验证：dataset 字段正确标注来源

- [ ] **T9.2.3** E6 SLO scale：config_1b + E6 + Tight
  - 验证：SLO 值确实被 scale 过

- [ ] **T9.2.4** analyze.py：用 smoke test 的结果跑所有图表函数，确认无报错、能生成 PDF

### T9.3 切换到 8B 前的 checklist

- [ ] 所有 T9.1 和 T9.2 通过
- [ ] 1B 的 calibration 报告合理
- [ ] E0_Smoke 的图表趋势合理（Our-System goodput >= No-FT under fault）
- [ ] 确认 8B 模型权重已下载
- [ ] 确认 8B profiling 完成且通过验证

---

## 依赖关系图

```
T0 (数据集) ───────────────────────────────────┐
                                                │
T1 (config 1b/8b/70b) ─────────────────────────┤
                                                ▼
T7 (profiling 脚本) ──→ T7.2 (跑 profiling) ──→ T3 (calibrate) ──→ T5 (suite) ──→ 跑实验
                                                ▲                                     │
T2 (workloads.py) ─────────────────────────────┤                                     │
                                                │                                     │
T4 (run.py) ───────────────────────────────────┘                                     │
                                                                                      ▼
T6 (analyze.py) ←──────────────────────────────────────────────── results_v2/ ──→ figures_v2/

T8 (run_all.sh) 串联以上全部
T9 (验证) 贯穿全过程，1B 先行验证
```

**可以并行做的**：
- T0 (数据集) + T1 (config) + T7.1 (profiling 脚本改动)：互不依赖
- T2 (workloads) + T4 (run.py)：互不依赖，但都依赖 T0 和 T1
- T6 (analyze)：可以在等实验跑的时候先写

**必须串行的**：
- T0 → T2（workloads 需要数据集文件）
- T1 + T7 → T3（calibrate 需要 config 和 profile）
- T3 → T5 → 跑实验 → T6
- **1B 全流程 → 8B 全流程 → 70B 全流程**

---

## 工作量预估

| Phase | 内容 | 预估时间 |
|---|---|---|
| Phase 0 | 数据集下载 + 预处理 + 验证 | 0.5 天 |
| Phase 1 | 3 个 config 文件 | 0.5 天 |
| Phase 2 | workloads.py 重写 | 1 天 |
| Phase 3 | calibrate.py 新建 | 1 天（代码） + 0.5 天（跑校准） |
| Phase 4 | run.py 改动 | 0.5 天 |
| Phase 5 | suite.py 改动 | 0.5 天 |
| Phase 6 | analyze.py 新增 6 种图表 + 多模型合并 | 1.5 天 |
| Phase 7 | Profiling 脚本改动 + 执行 | 0.5 天 |
| Phase 8 | run_all.sh | 0.25 天 |
| Phase 9 | 1B 验证 + 调试 | 1 天 |
| **合计代码工作** | | **~7.5 天** |
| **跑实验 (8B)** | | **~8 天**（串行，单 4-GPU 节点） |
| **跑实验 (70B)** | | **~2 天**（串行，单 8-GPU 节点） |
| **总计** | | **~17.5 天** |
