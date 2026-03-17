# LLM-Benchmark

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/yourusername/llm-benchmark/pulls)

一个专为自部署大模型设计的高性能负载压测工具，支持 vLLM、Ollama、Xinference、TGI、LMDeploy、SGLang 等主流部署方案。

## ✨ 特性

- **多部署方案支持**：一键配置 vLLM、Ollama、Xinference、TGI、LMDeploy、SGLang
- **双模式压测**：同步/异步并发，满足不同性能需求
- **流式响应支持**：精确测量 TTFT (Time To First Token) 和 TPS
- **持续压测**：按时间压测，模拟真实业务场景
- **GPU 监控**：实时监控 GPU 利用率、显存、温度等关键指标
- **模型预热**：自动预热避免冷启动偏差
- **丰富指标**：延迟统计(P50/P95/P99)、成功率、QPS、Token 吞吐量
- **灵活配置**：多种预设 Prompt、自定义参数、JSON 结果导出

## 🚀 快速开始

### 安装依赖

```bash
pip install aiohttp requests
```

### 基本用法

```bash
# 使用预设快速测试 vLLM
python llm_benchmark.py --preset vllm --model qwen2.5-7b

# 自定义参数压测
python llm_benchmark.py \
  --url http://localhost:8000/v1/chat/completions \
  --model qwen2.5-7b \
  --num-requests 100 \
  --concurrency 10 \
  --async-mode
```

## 📊 使用示例

### 1. 基础功能测试

```bash
python llm_benchmark.py \
  --preset vllm \
  --model qwen2.5-7b \
  --num-requests 10 \
  --concurrency 1 \
  --warmup
```

### 2. 高并发压测

```bash
python llm_benchmark.py \
  --preset vllm \
  --model qwen2.5-7b \
  --num-requests 100 \
  --concurrency 10 \
  --async-mode \
  --warmup \
  --monitor-gpu
```

### 3. 持续压测 60 秒

```bash
python llm_benchmark.py \
  --preset vllm \
  --model qwen2.5-7b \
  --duration 60 \
  --concurrency 5 \
  --async-mode \
  --monitor-gpu
```

### 4. 不同 Prompt 类型测试

```bash
# 短文本测试
python llm_benchmark.py --preset vllm --model qwen2.5-7b --prompt-type short

# 长文本测试
python llm_benchmark.py --preset vllm --model qwen2.5-7b --prompt-type long --max-tokens 1024

# 代码生成测试
python llm_benchmark.py --preset vllm --model qwen2.5-coder-7b --prompt-type code
```

## 🔧 支持的部署方案

| 方案 | 预设名 | 默认端口 | 启动命令示例 |
|------|--------|----------|--------------|
| vLLM | `vllm` | 8000 | `python -m vllm.entrypoints.openai.api_server --model qwen2.5-7b` |
| Ollama | `ollama` | 11434 | `ollama run llama3.1` |
| Xinference | `xinference` | 9997 | `xinference-local` |
| TGI | `tgi` | 8080 | `text-generation-launcher --model-id qwen/qwen2.5-7b` |
| LMDeploy | `lmdeploy` | 23333 | `lmdeploy serve api_server qwen2.5-7b` |
| SGLang | `sglang` | 30000 | `python -m sglang.launch_server --model qwen2.5-7b` |

## 📈 输出示例

```
======================================================================
大模型 API 负载压测报告
======================================================================
总请求数: 100
成功请求: 98
失败请求: 2
成功率: 98.00%
----------------------------------------------------------------------
延迟统计 (秒):
  平均延迟: 1.234s
  P50 延迟: 1.156s
  P95 延迟: 2.345s
  P99 延迟: 2.876s
  最小延迟: 0.876s
  最大延迟: 3.456s
----------------------------------------------------------------------
Token 统计:
  平均首Token时间 (TTFT): 0.123s
  平均生成速度 (TPS): 45.67 tokens/s
  总输入 Tokens: 2450
  总输出 Tokens: 45678
  总 Tokens: 48128
----------------------------------------------------------------------
GPU 统计:
  平均利用率: 87.5%
  峰值显存使用: 12288 MB
======================================================================

压测总耗时: 12.34s
QPS: 8.12
```

## 🛠️ 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `--url` | | API 地址 | 必填 |
| `--api-key` | | API 密钥 | `"EMPTY"` |
| `--model` | | 模型名称 | 必填 |
| `--preset` | | 预设配置 (vllm/ollama/xinference/tgi/lmdeploy/sglang) | 无 |
| `--prompt` | | 自定义测试 prompt | 无 |
| `--prompt-type` | | 预设 prompt 类型 (short/medium/long/code) | `"medium"` |
| `--num-requests` | `-n` | 总请求数 | `10` |
| `--concurrency` | `-c` | 并发数 | `1` |
| `--duration` | | 持续压测时间(秒) | 无 |
| `--max-tokens` | | 最大生成 token 数 | `512` |
| `--temperature` | | 温度参数 | `0.7` |
| `--timeout` | | 请求超时时间(秒) | `120` |
| `--async-mode` | `--async` | 使用异步模式 | False |
| `--warmup` | | 模型预热 | False |
| `--monitor-gpu` | | 监控 GPU 状态 | False |
| `--output` | | 结果输出 JSON 文件 | 无 |

## 📦 依赖

- Python 3.8+
- `aiohttp` - 异步 HTTP 客户端
- `requests` - 同步 HTTP 客户端

## 📄 输出格式

结果可以导出为 JSON 格式：

```json
{
  "config": {
    "url": "http://localhost:8000/v1/chat/completions",
    "model": "qwen2.5-7b",
    "num_requests": 100,
    "concurrency": 10,
    "max_tokens": 512,
    "temperature": 0.7,
    "async_mode": true
  },
  "results": {
    "total_requests": 100,
    "successful_requests": 98,
    "failed_requests": 2,
    "success_rate": 98.0,
    "avg_latency": 1.234,
    "p50_latency": 1.156,
    "p95_latency": 2.345,
    "p99_latency": 2.876,
    "avg_ttft": 0.123,
    "avg_tps": 45.67,
    "total_input_tokens": 2450,
    "total_output_tokens": 45678,
    "total_tokens": 48128,
    "total_time": 12.34,
    "qps": 8.12,
    "gpu": {
      "avg_util": 87.5,
      "peak_memory_mb": 12288
    }
  }
}
```

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

MIT License

## 💬 交流

如有问题或建议，欢迎提 Issue 或在 Discussion 中讨论。