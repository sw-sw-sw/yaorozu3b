import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

@dataclass
class QueueItem:
    data: Any
    timestamp: float
    delay: float
    process_func: Callable[[Any], None]

class DelayedQueue:
    def __init__(self):
        self.queue = deque()

    def add(self, data: Any, delay: float, process_func: Callable[[Any], None]):
        item = QueueItem(data, time.time(), delay, process_func)
        self.queue.append(item)

    def update(self):
        current_time = time.time()
        while self.queue and current_time >= self.queue[0].timestamp + self.queue[0].delay:
            item = self.queue.popleft()
            item.process_func(item.data)

'''
def main():
    delayed_queue = DelayedQueue()

    def process_item1(data):
        print(f"Processing item 1: {data}")

    def process_item2(data):
        print(f"Processing item 2: {data.upper()}")

    # メインループ
    try:
        # 異なる処理関数を持つアイテムを追加
        delayed_queue.add("hello", 2, process_item1)
        delayed_queue.add("world", 4, process_item2)

        start_time = time.time()
        while time.time() - start_time < 5:  # 5秒間実行
            delayed_queue.update()
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("プログラムを終了します")

if __name__ == "__main__":
    main()
    
'''