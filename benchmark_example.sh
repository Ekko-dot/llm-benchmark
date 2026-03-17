#!/bin/bash
# 大模型压测脚本使用示例 - 自部署模型版

# ========== 快速开始 ==========

# 1. 使用预设快速测试 vLLM (最常用)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  -n 10 \
  -c 1

# 2. 使用预设测试 Ollama
python llm_benchmark.py \
  --preset ollama \
  --model "llama3.1" \
  -n 10 \
  -c 1

# ========== 常用压测场景 ==========

# 3. 基础功能测试 (单并发，少量请求)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  -n 5 \
  -c 1 \
  --warmup \
  --output "basic_test.json"

# 4. 中等并发压测 (适合 7B 模型)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  -n 100 \
  -c 5 \
  --async \
  --warmup \
  --monitor-gpu

# 5. 高并发压测 (适合 13B+ 模型或多卡)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-14b" \
  -n 500 \
  -c 20 \
  --async \
  --warmup \
  --monitor-gpu \
  --output "high_concurrency.json"

# ========== 持续压测 (模拟真实负载) ==========

# 6. 持续压测 60 秒，5 并发
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  --duration 60 \
  -c 5 \
  --async \
  --monitor-gpu

# 7. 持续压测 5 分钟，10 并发
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  --duration 300 \
  -c 10 \
  --async \
  --monitor-gpu \
  --output "sustained_load.json"

# ========== 不同 Prompt 长度测试 ==========

# 8. 短文本测试 (快速响应)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  --prompt-type short \
  -n 50 \
  -c 5 \
  --async

# 9. 长文本测试 (考察长上下文处理能力)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  --prompt-type long \
  --max-tokens 1024 \
  -n 20 \
  -c 2 \
  --async \
  --monitor-gpu

# 10. 代码生成测试
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-coder-7b" \
  --prompt-type code \
  --max-tokens 1024 \
  -n 30 \
  -c 3 \
  --async

# ========== 不同部署方案测试 ==========

# 11. vLLM 压测
python llm_benchmark.py \
  --url "http://localhost:8000/v1/chat/completions" \
  --model "qwen2.5-7b" \
  -n 100 \
  -c 10 \
  --async \
  --warmup

# 12. Ollama 压测
python llm_benchmark.py \
  --url "http://localhost:11434/v1/chat/completions" \
  --model "llama3.1:8b" \
  -n 50 \
  -c 2 \
  --async

# 13. Xinference 压测
python llm_benchmark.py \
  --preset xinference \
  --model "custom-model-uid" \
  -n 100 \
  -c 5 \
  --async

# 14. TGI (Text Generation Inference) 压测
python llm_benchmark.py \
  --preset tgi \
  --model "tgi" \
  -n 100 \
  -c 5 \
  --async

# 15. LMDeploy 压测
python llm_benchmark.py \
  --preset lmdeploy \
  --model "qwen2.5-7b" \
  -n 100 \
  -c 10 \
  --async

# 16. SGLang 压测
python llm_benchmark.py \
  --preset sglang \
  --model "qwen2.5-7b" \
  -n 100 \
  -c 10 \
  --async

# ========== 极限压测 ==========

# 17. 超高并发压测 (需要多卡或高性能单卡)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  -n 1000 \
  -c 50 \
  --async \
  --warmup \
  --monitor-gpu \
  --timeout 300 \
  --output "extreme_load.json"

# 18. 长时间稳定性测试 (30分钟)
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  --duration 1800 \
  -c 10 \
  --async \
  --monitor-gpu \
  --output "stability_test.json"

# ========== 自定义参数示例 ==========

# 19. 自定义 prompt 和参数
python llm_benchmark.py \
  --preset vllm \
  --model "qwen2.5-7b" \
  --prompt "请用中文解释什么是量子计算，要求通俗易懂" \
  --max-tokens 2048 \
  --temperature 0.5 \
  -n 20 \
  -c 2 \
  --async

# 20. 测试不同温度参数的影响
for temp in 0.1 0.5 0.7 1.0; do
  echo "Testing temperature=$temp"
  python llm_benchmark.py \
    --preset vllm \
    --model "qwen2.5-7b" \
    --temperature $temp \
    -n 20 \
    -c 2 \
    --output "temp_${temp}.json"
done
