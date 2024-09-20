# class1.py
from TEST.performance_tracker import PerformanceTracker
import time

class Class1:
    @PerformanceTracker.measure_time
    def method1(self, n):
        time.sleep(0.1)  # シミュレーションのための遅延
        return sum(range(n))
