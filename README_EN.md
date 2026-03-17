# LLM-Benchmark

A high-performance load testing tool designed for self-deployed large language models, supporting mainstream deployment solutions like vLLM, Ollama, Xinference, TGI, LMDeploy, and SGLang.

## Features

- Multi-deployment support: One-click configuration for vLLM, Ollama, Xinference, TGI, LMDeploy, SGLang
- Dual-mode testing: Sync/async concurrency to meet different performance needs
- Streaming response support: Accurate measurement of TTFT (Time To First Token) and TPS
- Continuous testing: Time-based testing to simulate real business scenarios
- GPU monitoring: Real-time monitoring of GPU utilization, memory, temperature and other key metrics
- Model warm-up: Automatic warm-up to avoid cold start bias
- Rich metrics: Latency statistics (P50/P95/P99), success rate, QPS, Token throughput
- Flexible configuration: Multiple preset prompts, custom parameters, JSON result export

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```bash
# Quick test with preset for vLLM
python llm_benchmark.py --preset vllm --model qwen2.5-7b

# Custom parameters
python llm_benchmark.py \
  --url http://localhost:8000/v1/chat/completions \
  --model qwen2.5-7b \
  --num-requests 100 \
  --concurrency 10 \
  --async-mode
```

## Supported Deployment Solutions

| Solution | Preset | Port | Example Command |
|----------|--------|------|-----------------|
| vLLM | `vllm` | 8000 | `python -m vllm.entrypoints.openai.api_server --model qwen2.5-7b` |
| Ollama | `ollama` | 11434 | `ollama run llama3.1` |
| Xinference | `xinference` | 9997 | `xinference-local` |
| TGI | `tgi` | 8080 | `text-generation-launcher --model-id qwen/qwen2.5-7b` |
| LMDeploy | `lmdeploy` | 23333 | `lmdeploy serve api_server qwen2.5-7b` |
| SGLang | `sglang` | 30000 | `python -m sglang.launch_server --model qwen2.5-7b` |

## License

MIT License