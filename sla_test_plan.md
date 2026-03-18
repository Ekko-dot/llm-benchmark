# EvalScope SLA 自动调优测试方案

## 一、测试目标

使用 EvalScope 的 SLA 自动调优功能，通过二分查找算法自动寻找满足不同性能约束条件下的**最大并发数**，评估 Qwen2.5-7B-Instruct 模型在 LMDeploy 部署下的性能边界。

---

## 二、测试环境

| 项目 | 配置 |
|------|------|
| 机器 | 阿里云 PAI DSW |
| CPU | Intel Xeon Platinum 8369B @ 2.90GHz |
| 内存 | 29GB |
| GPU | NVIDIA A10 24GB VRAM |
| CUDA | 12.4 |
| 模型 | Qwen2.5-7B-Instruct |
| 推理引擎 | LMDeploy |
| 压测框架 | EvalScope |

---

## 三、测试场景

### 场景 1：延迟约束测试（P99 TTFT）

**目标**：寻找满足 P99 TTFT 阈值的最大并发数

| 测试项 | 约束条件 | 说明 |
|--------|----------|------|
| 1.1 | P99 TTFT <= 500ms | 严格延迟要求，适合实时对话场景 |
| 1.2 | P99 TTFT <= 1000ms | 中等延迟要求，适合一般交互场景 |
| 1.3 | P99 TTFT <= 2000ms | 宽松延迟要求，适合批处理场景 |

### 场景 2：ITL/TPOT 约束测试

**目标**：寻找满足 ITL（Inter-Token Latency）阈值的最大并发数

> **说明**：ITL 指相邻两个 token 之间的时间间隔。EvalScope 使用 TPOT (Time Per Output Token) 指标，TPOT = ITL 的平均值，可用于约束 ITL 性能。

| 测试项 | 约束条件 | 说明 |
|--------|----------|------|
| 2.1 | P99 TPOT <= 50ms | 严格 ITL 要求，适合实时流式输出场景 |
| 2.2 | P99 TPOT <= 100ms | 中等 ITL 要求，适合一般对话场景 |
| 2.3 | P99 TPOT <= 200ms | 宽松 ITL 要求，适合批处理场景 |

### 场景 3：吞吐量约束测试（TPS）

**目标**：寻找满足吞吐量阈值的最大并发数

| 测试项 | 约束条件 | 说明 |
|--------|----------|------|
| 3.1 | TPS >= 100 tok/s | 基础吞吐要求 |
| 3.2 | TPS >= 200 tok/s | 中等吞吐要求 |
| 3.3 | TPS >= 300 tok/s | 高吞吐要求 |

### 场景 4：组合约束测试

**目标**：寻找同时满足多个指标的平衡点

| 测试项 | 约束条件 | 说明 |
|--------|----------|------|
| 3.1 | P99 TTFT <= 1000ms AND TPS >= 150 tok/s | 平衡延迟与吞吐 |
| 3.2 | P99 TTFT <= 500ms AND TPS >= 100 tok/s | 偏重低延迟 |
| 3.3 | P99 TTFT <= 2000ms AND TPS >= 250 tok/s | 偏重高吞吐 |

---

## 四、测试步骤

### 4.1 环境准备

```bash
# 1. 确保 LMDeploy 服务已启动
lmdeploy serve api_server \
  /mnt/workspace/models/Qwen/Qwen2.5-7B-Instruct \
  --server-port 8000 \
  --tp 1

# 2. 验证服务可用
curl http://127.0.0.1:8000/v1/models
```

### 4.2 场景 1：延迟约束测试

```bash
# 测试 1.1: P99 TTFT <= 500ms
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=500"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 1.2: P99 TTFT <= 1000ms
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=1000"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 1.3: P99 TTFT <= 2000ms
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=2000"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test
```

### 4.3 场景 2：ITL/TPOT 约束测试

```bash
# 测试 2.1: P99 TPOT <= 50ms
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_tpot": "<=50"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 2.2: P99 TPOT <= 100ms
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_tpot": "<=100"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 2.3: P99 TPOT <= 200ms
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_tpot": "<=200"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test
```

### 4.4 场景 3：吞吐量约束测试

```bash
# 测试 3.1: TPS >= 100 tok/s
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"output_tps": ">=100"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 3.2: TPS >= 200 tok/s
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"output_tps": ">=200"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 3.3: TPS >= 300 tok/s
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"output_tps": ">=300"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test
```

### 4.5 场景 4：组合约束测试

```bash
# 测试 4.1: P99 TTFT <= 1000ms AND TPS >= 150 tok/s
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=1000", "output_tps": ">=150"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 4.2: P99 TTFT <= 500ms AND TPS >= 100 tok/s
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=500", "output_tps": ">=100"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 4.3: P99 TTFT <= 2000ms AND TPS >= 250 tok/s
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=2000", "output_tps": ">=250"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test

# 测试 4.4: P99 TTFT <= 1000ms AND P99 TPOT <= 100ms (延迟+ITL组合)
CUDA_VISIBLE_DEVICES="" evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --model Qwen2.5-7B-Instruct \
  --api openai \
  --dataset openqa \
  --stream \
  --max-tokens 512 \
  --sla-auto-tune \
  --sla-variable parallel \
  --sla-params '[{"p99_ttft": "<=1000", "p99_tpot": "<=100"}]' \
  --sla-num-runs 3 \
  --sla-lower-bound 1 \
  --sla-upper-bound 50 \
  --outputs-dir /mnt/workspace/outputs/sla_test
```

---

## 五、参数说明

| 参数 | 说明 | 本次配置 |
|------|------|----------|
| `--sla-auto-tune` | 启用 SLA 自动调优模式 | 开启 |
| `--sla-variable` | 调优变量 | parallel（并发数） |
| `--sla-params` | SLA 约束条件，JSON 格式 | 根据场景配置 |
| `--sla-num-runs` | 每个配置运行次数（取平均） | 3 次 |
| `--sla-lower-bound` | 并发数搜索下限 | 1 |
| `--sla-upper-bound` | 并发数搜索上限 | 50 |
| `--max-tokens` | 最大输出 token 数 | 512 |
| `--stream` | 流式输出（必须开启才能测 TTFT） | 开启 |

### SLA 指标说明

| 指标 | 含义 | 约束格式 |
|------|------|----------|
| `p99_ttft` | 99% 请求的首 token 延迟 (TTFT) | `"<=X"` (毫秒) |
| `avg_ttft` | 平均首 token 延迟 | `"<=X"` (毫秒) |
| `p99_tpot` | 99% 的 token 输出间隔 (ITL) | `"<=X"` (毫秒) |
| `avg_tpot` | 平均 token 输出间隔 | `"<=X"` (毫秒) |
| `p99_latency` | 99% 请求的总延迟 | `"<=X"` (毫秒) |
| `output_tps` | 输出吞吐量 | `">=X"` (tok/s) |
| `total_tps` | 总吞吐量（输入+输出） | `">=X"` (tok/s) |
| `rps` | 每秒请求数 | `">=X"` |

---

## 六、预期输出

每个测试将输出：

1. **最优并发数**：满足 SLA 约束的最大并发数
2. **性能报告**：HTML 可视化报告
3. **详细数据**：SQLite 数据库文件
4. **摘要信息**：包含关键指标的性能摘要

---

## 七、执行顺序

```
1. 启动 LMDeploy 服务
2. 执行场景 1（TTFT 延迟约束）- 3 个测试
3. 执行场景 2（ITL/TPOT 约束）- 3 个测试
4. 执行场景 3（吞吐量约束）- 3 个测试
5. 执行场景 4（组合约束）- 4 个测试
6. 汇总分析所有结果
```

---

## 八、注意事项

1. **服务稳定性**：每次测试前确认 LMDeploy 服务正常运行
2. **显存监控**：高并发时注意 GPU 显存使用，避免 OOM
3. **测试间隔**：各测试之间建议间隔 10-30 秒，让服务恢复稳定
4. **结果对比**：建议将所有测试结果输出到同一目录便于对比分析

---

## 九、结果分析模板

测试完成后，将汇总以下内容：

| 测试项 | SLA 约束 | 最优并发数 | 实际 P99 TTFT | 实际 P99 TPOT | 实际 TPS |
|--------|----------|------------|---------------|---------------|----------|
| 1.1 | P99 TTFT <= 500ms | - | - | - | - |
| 1.2 | P99 TTFT <= 1000ms | - | - | - | - |
| 1.3 | P99 TTFT <= 2000ms | - | - | - | - |
| 2.1 | P99 TPOT <= 50ms | - | - | - | - |
| 2.2 | P99 TPOT <= 100ms | - | - | - | - |
| 2.3 | P99 TPOT <= 200ms | - | - | - | - |
| 3.1 | TPS >= 100 tok/s | - | - | - | - |
| 3.2 | TPS >= 200 tok/s | - | - | - | - |
| 3.3 | TPS >= 300 tok/s | - | - | - | - |
| 4.1 | TTFT<=1000ms & TPS>=150 | - | - | - | - |
| 4.2 | TTFT<=500ms & TPS>=100 | - | - | - | - |
| 4.3 | TTFT<=2000ms & TPS>=250 | - | - | - | - |
| 4.4 | TTFT<=1000ms & TPOT<=100ms | - | - | - | - |
