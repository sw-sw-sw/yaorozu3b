# performance_tracker.py
import time
from functools import wraps
from collections import defaultdict
import threading

class PerformanceTracker:
    _instance = None
    execution_times = defaultdict(list)
    update_interval = 1
    stop_flag = threading.Event()
    lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def measure_time(cls, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            with cls.lock:
                cls.execution_times[func.__name__].append(execution_time)
            return result
        return wrapper

    @classmethod
    def print_stats(cls):
        with cls.lock:
            print("\n現在の性能統計:")
            for func_name, times in cls.execution_times.items():
                if times:
                    avg_time = sum(times) / len(times)
                    min_time = min(times)
                    max_time = max(times)
                    print(f"{func_name}:")
                    print(f"  平均実行時間: {avg_time:.6f}秒")
                    print(f"  最小実行時間: {min_time:.6f}秒")
                    print(f"  最大実行時間: {max_time:.6f}秒")
                    print(f"  呼び出し回数: {len(times)}")
            print("\n")

    @classmethod
    def start_real_time_tracking(cls, interval=1):
        cls.update_interval = interval
        def update_stats():
            while not cls.stop_flag.is_set():
                cls.print_stats()
                time.sleep(cls.update_interval)

        cls.tracking_thread = threading.Thread(target=update_stats)
        cls.tracking_thread.start()

    @classmethod
    def stop_real_time_tracking(cls):
        cls.stop_flag.set()
        if hasattr(cls, 'tracking_thread'):
            cls.tracking_thread.join()