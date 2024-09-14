# pythonのバージョン違いによる処理速度テスト

| Column 1  | benchmark1  |
|:----------|:----------|
| python10  | 1.8248 seconds    |    
| python11  | 1.0752 seconds    |     |  
| pypy3 python10  | 0.0835 seconds    |   
| pypy3 python11  | 0.0807 seconds    |  


## Benchmark1

```python

import timedef fibonacci(n):    if n <= 1:        return n    else:        return fibonacci(n-1) + fibonacci(n-2)def main():    start_time = time.time()        result = fibonacci(35)        end_time = time.time()    execution_time = end_time - start_time        print(f"Fibonacci(35) = {result}")    print(f"Execution time: {execution_time:.4f} seconds")if __name__ == "__main__":    main()

```

## Benchmark2

```python
import timeimport random# ベンチマーク用の関数を定義def benchmark_cpu_bound_operations():    print("CPU bound operation (calculating squares):")    large_list = list(range(10**7))  # 1千万個の整数リストを作成    start_time = time.time()        # 各要素の平方を計算    squares = [x**2 for x in large_list]        end_time = time.time()    print(f"Time taken: {end_time - start_time:.5f} seconds")    return end_time - start_timedef benchmark_memory_operations():    print("Memory bound operation (sorting a large list):")    large_list = [random.random() for _ in range(10**7)]  # 1千万個のランダムな浮動小数点リストを作成    start_time = time.time()        # リストをソート    large_list.sort()        end_time = time.time()    print(f"Time taken: {end_time - start_time:.5f} seconds")    return end_time - start_time# ベンチマークを実行cpu_time = benchmark_cpu_bound_operations()memory_time = benchmark_memory_operations()# 結果を表示print(f"\nSummary:\nCPU-bound operation time: {cpu_time:.5f} seconds")print(f"Memory-bound operation time: {memory_time:.5f} seconds")
```