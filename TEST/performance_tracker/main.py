# main.py
from TEST.performance_tracker import PerformanceTracker
from class1 import Class1
from class2 import Class2
import time

def main():
    PerformanceTracker.start_real_time_tracking(interval=1)

    obj1 = Class1()
    obj2 = Class2()

    try:
        for _ in range(10):
            obj1.method1(10000)
            obj2.method2(1000)
            time.sleep(0.5)  # 処理間隔を設定
    finally:
        PerformanceTracker.stop_real_time_tracking()

    print("テスト終了")

if __name__ == "__main__":
    main()