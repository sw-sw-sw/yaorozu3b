# class2.py
from TEST.performance_tracker import PerformanceTracker
import time
class Class2:
    @PerformanceTracker.measure_time
    def method2(self, n):
        time.sleep(0.2)  # シミュレーションのための遅延
        return [i**2 for i in range(n)]
