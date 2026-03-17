#!/usr/bin/env python3
"""
大模型 API 负载压测脚本 (自部署模型优化版)
支持 vLLM、Ollama、Xinference、TGI、LMDeploy 等自部署方案
"""

import asyncio
import json
import time
import statistics
import subprocess
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import argparse
import sys

import aiohttp
import requests


# ============ 预设配置 ============
PRESETS = {
    "vllm": {
        "url": "http://localhost:8000/v1/chat/completions",
        "api_key": "EMPTY",
    },
    "ollama": {
        "url": "http://localhost:11434/v1/chat/completions",
        "api_key": "EMPTY",
    },
    "xinference": {
        "url": "http://localhost:9997/v1/chat/completions",
        "api_key": "EMPTY",
    },
    "tgi": {
        "url": "http://localhost:8080/v1/chat/completions",
        "api_key": "EMPTY",
    },
    "lmdeploy": {
        "url": "http://localhost:23333/v1/chat/completions",
        "api_key": "EMPTY",
    },
    "sglang": {
        "url": "http://localhost:30000/v1/chat/completions",
        "api_key": "EMPTY",
    },
}

# 不同长度的测试 Prompts
TEST_PROMPTS = {
    "short": "你好，请简单介绍一下自己。",
    "medium": "请详细介绍一下人工智能的发展历程，包括重要的里程碑事件。",
    "long": """请详细解释深度学习中的 Transformer 架构，包括：
1. 自注意力机制（Self-Attention）的原理
2. 多头注意力（Multi-Head Attention）的作用
3. 位置编码（Positional Encoding）的实现方式
4. 编码器和解码器的结构差异
5. 在 NLP 任务中的应用案例

请用通俗易懂的语言，配合适当的例子进行说明。""",
    "code": """请编写一个 Python 函数，实现快速排序算法，要求：
1. 使用递归实现
2. 添加类型注解
3. 包含详细的文档字符串
4. 提供测试用例
5. 分析时间复杂度""",
}


@dataclass
class RequestResult:
    """单个请求的结果"""
    success: bool
    latency: float  # 总延迟（秒）
    ttft: float  # Time To First Token（秒）
    tps: float  # Tokens Per Second
    input_tokens: int
    output_tokens: int
    error_message: Optional[str] = None


@dataclass
class GPUStats:
    """GPU 统计信息"""
    timestamp: float
    gpu_util: float  # GPU 利用率 %
    memory_used: int  # MB
    memory_total: int  # MB
    temperature: float  # °C
    power_draw: float  # W


@dataclass
class BenchmarkStats:
    """压测统计结果"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    latencies: List[float] = field(default_factory=list)
    ttfts: List[float] = field(default_factory=list)
    tpss: List[float] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    gpu_stats: List[GPUStats] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def p50_latency(self) -> float:
        return statistics.median(self.latencies) if self.latencies else 0.0

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency(self) -> float:
        if not self.latencies:
            return 0.0
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def avg_ttft(self) -> float:
        return statistics.mean(self.ttfts) if self.ttfts else 0.0

    @property
    def avg_tps(self) -> float:
        return statistics.mean(self.tpss) if self.tpss else 0.0

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def avg_gpu_util(self) -> float:
        if not self.gpu_stats:
            return 0.0
        return statistics.mean([g.gpu_util for g in self.gpu_stats])

    @property
    def peak_gpu_memory(self) -> int:
        if not self.gpu_stats:
            return 0
        return max([g.memory_used for g in self.gpu_stats])

    def print_report(self):
        """打印压测报告"""
        print("\n" + "=" * 70)
        print("大模型 API 负载压测报告")
        print("=" * 70)
        print(f"总请求数: {self.total_requests}")
        print(f"成功请求: {self.successful_requests}")
        print(f"失败请求: {self.failed_requests}")
        print(f"成功率: {self.success_rate:.2f}%")
        print("-" * 70)
        print("延迟统计 (秒):")
        print(f"  平均延迟: {self.avg_latency:.3f}s")
        print(f"  P50 延迟: {self.p50_latency:.3f}s")
        print(f"  P95 延迟: {self.p95_latency:.3f}s")
        print(f"  P99 延迟: {self.p99_latency:.3f}s")
        print(f"  最小延迟: {min(self.latencies):.3f}s" if self.latencies else "  最小延迟: N/A")
        print(f"  最大延迟: {max(self.latencies):.3f}s" if self.latencies else "  最大延迟: N/A")
        print("-" * 70)
        print("Token 统计:")
        print(f"  平均首Token时间 (TTFT): {self.avg_ttft:.3f}s")
        print(f"  平均生成速度 (TPS): {self.avg_tps:.2f} tokens/s")
        print(f"  总输入 Tokens: {self.total_input_tokens}")
        print(f"  总输出 Tokens: {self.total_output_tokens}")
        print(f"  总 Tokens: {self.total_tokens}")
        if self.gpu_stats:
            print("-" * 70)
            print("GPU 统计:")
            print(f"  平均利用率: {self.avg_gpu_util:.1f}%")
            print(f"  峰值显存使用: {self.peak_gpu_memory} MB")
        print("=" * 70)


class GPUMonitor:
    """GPU 监控器"""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.running = False
        self.stats: List[GPUStats] = []
        self.thread: Optional[threading.Thread] = None

    def _collect(self):
        """收集 GPU 状态"""
        while self.running:
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 5:
                        stat = GPUStats(
                            timestamp=time.time(),
                            gpu_util=float(parts[0]),
                            memory_used=int(parts[1]),
                            memory_total=int(parts[2]),
                            temperature=float(parts[3]),
                            power_draw=float(parts[4]),
                        )
                        self.stats.append(stat)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        """开始监控"""
        self.running = True
        self.thread = threading.Thread(target=self._collect)
        self.thread.daemon = True
        self.thread.start()

    def stop(self) -> List[GPUStats]:
        """停止监控并返回统计"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        return self.stats


class LLMBenchmark:
    """大模型压测器"""

    def __init__(
        self,
        api_url: str,
        api_key: str,
        model: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        timeout: int = 120,
    ):
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def warmup(self, num_requests: int = 3):
        """模型预热"""
        print(f"进行模型预热 ({num_requests} 次请求)...")
        for i in range(num_requests):
            try:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": "你好"}],
                    "max_tokens": 50,
                    "temperature": 0.7,
                }
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                print(f"  预热请求 {i+1}/{num_requests} 完成")
            except Exception as e:
                print(f"  预热请求 {i+1}/{num_requests} 失败: {e}")
        print("预热完成\n")

    def _make_request(self) -> RequestResult:
        """发送单个同步请求"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        start_time = time.time()
        first_token_time = None
        output_text = ""

        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
                stream=True,
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    output_text += content
                        except json.JSONDecodeError:
                            continue

            end_time = time.time()
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else total_time

            # 估算 token 数 (简单估算: 1 token ≈ 4 字符)
            input_tokens = len(self.prompt) // 4
            output_tokens = len(output_text) // 4
            tps = output_tokens / total_time if total_time > 0 else 0

            return RequestResult(
                success=True,
                latency=total_time,
                ttft=ttft,
                tps=tps,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            return RequestResult(
                success=False,
                latency=time.time() - start_time,
                ttft=0,
                tps=0,
                input_tokens=0,
                output_tokens=0,
                error_message=str(e),
            )

    async def _async_request(self, session: aiohttp.ClientSession) -> RequestResult:
        """发送单个异步请求"""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self.prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": True,
        }

        start_time = time.time()
        first_token_time = None
        output_text = ""

        try:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            ) as response:
                response.raise_for_status()

                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if chunk.get("choices"):
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    if first_token_time is None:
                                        first_token_time = time.time()
                                    output_text += content
                        except json.JSONDecodeError:
                            continue

            end_time = time.time()
            total_time = end_time - start_time
            ttft = first_token_time - start_time if first_token_time else total_time

            input_tokens = len(self.prompt) // 4
            output_tokens = len(output_text) // 4
            tps = output_tokens / total_time if total_time > 0 else 0

            return RequestResult(
                success=True,
                latency=total_time,
                ttft=ttft,
                tps=tps,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            return RequestResult(
                success=False,
                latency=time.time() - start_time,
                ttft=0,
                tps=0,
                input_tokens=0,
                output_tokens=0,
                error_message=str(e),
            )

    def run_sync_benchmark(
        self, num_requests: int, concurrency: int, monitor: Optional[GPUMonitor] = None
    ) -> BenchmarkStats:
        """运行同步压测"""
        stats = BenchmarkStats()

        print(f"开始同步压测: {num_requests} 个请求, 并发数 {concurrency}")
        print("-" * 70)

        if monitor:
            monitor.start()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(self._make_request) for _ in range(num_requests)
            ]

            for i, future in enumerate(futures):
                result = future.result()
                stats.total_requests += 1

                if result.success:
                    stats.successful_requests += 1
                    stats.latencies.append(result.latency)
                    stats.ttfts.append(result.ttft)
                    stats.tpss.append(result.tps)
                    stats.total_input_tokens += result.input_tokens
                    stats.total_output_tokens += result.output_tokens
                    print(f"[{i+1}/{num_requests}] 成功 - 延迟: {result.latency:.3f}s, TTFT: {result.ttft:.3f}s, TPS: {result.tps:.1f}")
                else:
                    stats.failed_requests += 1
                    print(f"[{i+1}/{num_requests}] 失败 - 错误: {result.error_message}")

        if monitor:
            stats.gpu_stats = monitor.stop()

        return stats

    async def run_async_benchmark(
        self, num_requests: int, concurrency: int, monitor: Optional[GPUMonitor] = None
    ) -> BenchmarkStats:
        """运行异步压测"""
        stats = BenchmarkStats()
        semaphore = asyncio.Semaphore(concurrency)

        print(f"开始异步压测: {num_requests} 个请求, 并发数 {concurrency}")
        print("-" * 70)

        if monitor:
            monitor.start()

        async def bounded_request(session, idx):
            async with semaphore:
                result = await self._async_request(session)
                return idx, result

        async with aiohttp.ClientSession() as session:
            tasks = [
                bounded_request(session, i) for i in range(num_requests)
            ]

            for coro in asyncio.as_completed(tasks):
                idx, result = await coro
                stats.total_requests += 1

                if result.success:
                    stats.successful_requests += 1
                    stats.latencies.append(result.latency)
                    stats.ttfts.append(result.ttft)
                    stats.tpss.append(result.tps)
                    stats.total_input_tokens += result.input_tokens
                    stats.total_output_tokens += result.output_tokens
                    print(f"[{idx+1}/{num_requests}] 成功 - 延迟: {result.latency:.3f}s, TTFT: {result.ttft:.3f}s, TPS: {result.tps:.1f}")
                else:
                    stats.failed_requests += 1
                    print(f"[{idx+1}/{num_requests}] 失败 - 错误: {result.error_message}")

        if monitor:
            stats.gpu_stats = monitor.stop()

        return stats


def run_continuous_benchmark(
    benchmark: LLMBenchmark,
    duration: int,
    concurrency: int,
    async_mode: bool,
    monitor_gpu: bool,
):
    """运行持续压测（按时间而非请求数）"""
    stats = BenchmarkStats()
    semaphore = asyncio.Semaphore(concurrency) if async_mode else None
    monitor = GPUMonitor() if monitor_gpu else None

    print(f"开始持续压测: 持续时间 {duration} 秒, 并发数 {concurrency}")
    print("-" * 70)

    if monitor:
        monitor.start()

    start_time = time.time()
    request_count = 0

    async def async_worker(session):
        nonlocal request_count
        while time.time() - start_time < duration:
            async with semaphore:
                result = await benchmark._async_request(session)
                request_count += 1
                stats.total_requests += 1

                if result.success:
                    stats.successful_requests += 1
                    stats.latencies.append(result.latency)
                    stats.ttfts.append(result.ttft)
                    stats.tpss.append(result.tps)
                    stats.total_input_tokens += result.input_tokens
                    stats.total_output_tokens += result.output_tokens
                    print(f"[{request_count}] 成功 - 延迟: {result.latency:.3f}s")
                else:
                    stats.failed_requests += 1
                    print(f"[{request_count}] 失败 - 错误: {result.error_message}")

    async def run_async():
        async with aiohttp.ClientSession() as session:
            tasks = [async_worker(session) for _ in range(concurrency)]
            await asyncio.gather(*tasks)

    def sync_worker():
        nonlocal request_count
        while time.time() - start_time < duration:
            result = benchmark._make_request()
            request_count += 1
            stats.total_requests += 1

            if result.success:
                stats.successful_requests += 1
                stats.latencies.append(result.latency)
                stats.ttfts.append(result.ttft)
                stats.tpss.append(result.tps)
                stats.total_input_tokens += result.input_tokens
                stats.total_output_tokens += result.output_tokens
                print(f"[{request_count}] 成功 - 延迟: {result.latency:.3f}s")
            else:
                stats.failed_requests += 1
                print(f"[{request_count}] 失败 - 错误: {result.error_message}")

    if async_mode:
        asyncio.run(run_async())
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(sync_worker) for _ in range(concurrency)]
            for f in futures:
                f.result()

    if monitor:
        stats.gpu_stats = monitor.stop()

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="大模型 API 负载压测工具 (自部署模型优化版)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用预设快速测试 vLLM
  python llm_benchmark.py --preset vllm --model qwen2.5-7b

  # 自定义参数压测
  python llm_benchmark.py --url http://localhost:8000/v1/chat/completions --model qwen2.5-7b -n 100 -c 10 --async

  # 持续压测 60 秒
  python llm_benchmark.py --preset vllm --model qwen2.5-7b --duration 60 -c 5 --async

  # 使用长文本 prompt 测试
  python llm_benchmark.py --preset vllm --model qwen2.5-7b --prompt-type long -n 50 -c 5
        """
    )

    # 基础配置
    parser.add_argument("--url", help="API 地址")
    parser.add_argument("--api-key", default="EMPTY", help="API Key")
    parser.add_argument("--model", help="模型名称")
    parser.add_argument("--preset", choices=PRESETS.keys(), help="使用预设配置 (vllm/ollama/xinference/tgi/lmdeploy/sglang)")

    # Prompt 配置
    parser.add_argument("--prompt", help="自定义测试 prompt")
    parser.add_argument("--prompt-type", choices=TEST_PROMPTS.keys(), default="medium", help="使用预设 prompt 类型")

    # 压测参数
    parser.add_argument("-n", "--num-requests", type=int, default=10, help="总请求数 (默认: 10)")
    parser.add_argument("-c", "--concurrency", type=int, default=1, help="并发数 (默认: 1)")
    parser.add_argument("--duration", type=int, help="持续压测时间(秒)，设置后忽略 -n")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数 (默认: 512)")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数 (默认: 0.7)")
    parser.add_argument("--timeout", type=int, default=120, help="请求超时时间(秒) (默认: 120)")

    # 功能开关
    parser.add_argument("--async", dest="async_mode", action="store_true", help="使用异步模式")
    parser.add_argument("--warmup", action="store_true", help="压测前进行模型预热")
    parser.add_argument("--monitor-gpu", action="store_true", help="监控 GPU 状态")
    parser.add_argument("--output", help="结果输出到 JSON 文件")

    args = parser.parse_args()

    # 应用预设配置
    if args.preset:
        preset = PRESETS[args.preset]
        if not args.url:
            args.url = preset["url"]
        if args.api_key == "EMPTY":
            args.api_key = preset["api_key"]
        print(f"使用预设: {args.preset}")
        print(f"  API 地址: {args.url}")

    if not args.url or not args.model:
        parser.error("必须提供 --url 和 --model 参数，或使用 --preset")

    # 确定 prompt
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = TEST_PROMPTS[args.prompt_type]
        print(f"使用预设 prompt 类型: {args.prompt_type}")

    benchmark = LLMBenchmark(
        api_url=args.url,
        api_key=args.api_key,
        model=args.model,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    # 模型预热
    if args.warmup:
        benchmark.warmup()

    # 创建 GPU 监控器
    monitor = GPUMonitor() if args.monitor_gpu else None

    # 运行压测
    start_time = time.time()

    if args.duration:
        # 持续压测模式
        stats = run_continuous_benchmark(
            benchmark=benchmark,
            duration=args.duration,
            concurrency=args.concurrency,
            async_mode=args.async_mode,
            monitor_gpu=args.monitor_gpu,
        )
    else:
        # 固定请求数模式
        if args.async_mode:
            stats = asyncio.run(
                benchmark.run_async_benchmark(args.num_requests, args.concurrency, monitor)
            )
        else:
            stats = benchmark.run_sync_benchmark(args.num_requests, args.concurrency, monitor)

    total_time = time.time() - start_time

    # 打印报告
    stats.print_report()
    print(f"\n压测总耗时: {total_time:.2f}s")
    print(f"QPS: {stats.successful_requests / total_time:.2f}")

    # 保存结果
    if args.output:
        result_data = {
            "config": {
                "url": args.url,
                "model": args.model,
                "num_requests": args.num_requests if not args.duration else None,
                "duration": args.duration,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "async_mode": args.async_mode,
            },
            "results": {
                "total_requests": stats.total_requests,
                "successful_requests": stats.successful_requests,
                "failed_requests": stats.failed_requests,
                "success_rate": stats.success_rate,
                "avg_latency": stats.avg_latency,
                "p50_latency": stats.p50_latency,
                "p95_latency": stats.p95_latency,
                "p99_latency": stats.p99_latency,
                "avg_ttft": stats.avg_ttft,
                "avg_tps": stats.avg_tps,
                "total_input_tokens": stats.total_input_tokens,
                "total_output_tokens": stats.total_output_tokens,
                "total_tokens": stats.total_tokens,
                "total_time": total_time,
                "qps": stats.successful_requests / total_time,
            },
        }
        if stats.gpu_stats:
            result_data["results"]["gpu"] = {
                "avg_util": stats.avg_gpu_util,
                "peak_memory_mb": stats.peak_gpu_memory,
            }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
