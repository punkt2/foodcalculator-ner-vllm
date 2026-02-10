#!/usr/bin/env python3
"""
vLLM inference script for foodcalculator-nutrition model on Jetson.

Usage (inside Docker container):
    python vllm_inference.py --prompt "Calcium: 3.0% Phosphor:"
    python vllm_inference.py --interactive
    python vllm_inference.py --benchmark
    python vllm_inference.py --quantize --max_model_len 2048
"""

import argparse
import time
from vllm import LLM, SamplingParams


def load_model(
    model_name: str = "punkt2/foodcalculator-nutrition-SmolLM2-360M",
    gpu_memory_utilization: float = 0.6,
    max_model_len: int = 512,
    quantize: bool = False,
):
    """Load vLLM model optimized for Jetson."""
    print(f"Loading {model_name}...")
    if quantize:
        print("Using bitsandbytes 4-bit quantization")
    print(f"Max model length: {max_model_len}")
    
    start = time.perf_counter()
    
    kwargs = dict(
        model=model_name,
        dtype="float16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )
    
    if quantize:
        kwargs["quantization"] = "bitsandbytes"
        kwargs["load_format"] = "bitsandbytes"
    
    llm = LLM(**kwargs)
    
    elapsed = time.perf_counter() - start
    print(f"Model loaded in {elapsed:.1f}s")
    return llm


def generate(
    llm: LLM,
    prompts: list[str],
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> list[str]:
    """Generate text from prompts."""
    params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
    )
    
    outputs = llm.generate(prompts, params)
    return [output.outputs[0].text for output in outputs]


def benchmark(llm: LLM, num_runs: int = 10, max_tokens: int = 50):
    """Benchmark inference speed."""
    prompts = [
        "Calcium: 3.0% Phosphor:",
        "Analytische Bestandteile: Rohprotein 24,0%, Rohfett 14,0%, Rohfaser 2,5%, Rohasche",
        "Zusatzstoffe je kg: Vitamin A 15.000 IE, Vitamin D3 1.500 IE, Vitamin E",
        "Zusammensetzung: Geflügelfleisch, Reis, Tierfett. Analytische Bestandteile: Rohprotein 26%, Rohfett 16%, Rohfaser 2,5%, Rohasche 7%, Calcium 1,4%, Phosphor",
        "Energie 1450 kcal/kg, Protein 12.5g, Fett 8.2g, Kohlenhydrate 65.0g, Ballaststoffe",
    ]
    params = SamplingParams(max_tokens=max_tokens, temperature=0)

    print(f"\nBenchmarking with {num_runs} runs, {max_tokens} tokens each...")
    print(f"Prompts: {len(prompts)} (varying lengths)")
    print("=" * 60)

    for _ in range(3):
        _ = llm.generate(prompts[:1], params)

    total_tokens = 0
    total_time = 0

    for i, prompt in enumerate(prompts):
        times = []
        run_tokens = []
        for _ in range(num_runs):
            start = time.perf_counter()
            outputs = llm.generate([prompt], params)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            run_tokens.append(len(outputs[0].outputs[0].token_ids))

        avg_time = sum(times) / len(times)
        avg_tokens = sum(run_tokens) / len(run_tokens)
        tokens_per_sec = avg_tokens / avg_time
        min_time = min(times)
        max_time = max(times)
        total_tokens += sum(run_tokens)
        total_time += sum(times)

        label = prompt[:60] + "..." if len(prompt) > 60 else prompt
        print(f"\nPrompt {i+1}: \"{label}\"")
        print(f"  Output: \"{outputs[0].outputs[0].text.strip()[:60]}...\"")
        print(f"  Tokens: {avg_tokens:.0f} avg, Time: {avg_time*1000:.1f}ms avg ({min_time*1000:.1f}-{max_time*1000:.1f}ms)")
        print(f"  Speed: {tokens_per_sec:.1f} tokens/sec")

    overall_speed = total_tokens / total_time
    print("\n" + "=" * 60)
    print(f"Overall: {overall_speed:.1f} tokens/sec ({total_tokens} tokens in {total_time:.1f}s)")


def interactive(llm: LLM, max_tokens: int = 100):
    """Interactive mode for testing prompts."""
    print("\nInteractive mode (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\nPrompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue
            
            start = time.perf_counter()
            results = generate(llm, [prompt], max_tokens=max_tokens)
            elapsed = time.perf_counter() - start
            
            print(f"Output: {prompt}{results[0]}")
            print(f"Time: {elapsed*1000:.1f}ms")
            
        except KeyboardInterrupt:
            break
    
    print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description="vLLM inference for Jetson")
    parser.add_argument("--model", type=str, 
                        default="punkt2/foodcalculator-nutrition-SmolLM2-360M")
    parser.add_argument("--prompt", type=str, help="Single prompt to generate")
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--interactive", action="store_true", 
                        help="Interactive mode")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark")
    parser.add_argument("--gpu_memory", type=float, default=0.6,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max_model_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--quantize", action="store_true",
                        help="Use bitsandbytes 4-bit quantization (less memory, slightly slower)")
    
    args = parser.parse_args()
    
    llm = load_model(
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        quantize=args.quantize,
    )
    
    if args.benchmark:
        benchmark(llm, max_tokens=args.max_tokens)
    elif args.interactive:
        interactive(llm, max_tokens=args.max_tokens)
    elif args.prompt:
        results = generate(
            llm, [args.prompt], 
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"{args.prompt}{results[0]}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
