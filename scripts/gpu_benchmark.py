import torch
import time

def benchmark_compute(size=4096):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on {device}...")
    
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warmup
    for _ in range(5): torch.matmul(a, b)

    if torch.cuda.is_available(): torch.cuda.synchronize()
    start = time.time()
    
    iters = 20
    for _ in range(iters): torch.matmul(a, b)
    
    if torch.cuda.is_available(): torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iters
    tflops = (2 * size**3) / (avg_time * 1e12)
    print(f"Matrix {size}x{size} | Avg Time: {avg_time:.4f}s | Performance: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    benchmark_compute()