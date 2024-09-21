import unittest
import time
from unittest.mock import Mock
from delayed_queue import DelayedQueue  # 元のコードが delayed_queue.py に保存されていると仮定

class TestDelayedQueue(unittest.TestCase):
    def setUp(self):
        self.queue = DelayedQueue()

    def test_add_and_process_item(self):
        mock_process = Mock()
        self.queue.add("test_item", 0.1)
        
        # 処理されていないことを確認
        self.queue.update(mock_process)
        mock_process.assert_not_called()
        
        # 0.1秒待機
        time.sleep(0.11)
        
        # 処理されたことを確認
        self.queue.update(mock_process)
        mock_process.assert_called_once_with("test_item")

    def test_multiple_items(self):
        mock_process = Mock()
        self.queue.add("item1", 0.1)
        self.queue.add("item2", 0.2)
        self.queue.add("item3", 0.3)
        
        # 0.15秒待機
        time.sleep(0.15)
        self.queue.update(mock_process)
        mock_process.assert_called_once_with("item1")
        mock_process.reset_mock()
        
        # さらに0.1秒待機
        time.sleep(0.1)
        self.queue.update(mock_process)
        mock_process.assert_called_once_with("item2")
        mock_process.reset_mock()
        
        # さらに0.1秒待機
        time.sleep(0.1)
        self.queue.update(mock_process)
        mock_process.assert_called_once_with("item3")

    def test_empty_queue(self):
        mock_process = Mock()
        self.queue.update(mock_process)
        mock_process.assert_not_called()

    def test_order_of_processing(self):
        mock_process = Mock()
        self.queue.add("item2", 0.2)
        self.queue.add("item1", 0.1)
        self.queue.add("item3", 0.3)
        
        time.sleep(0.25)
        self.queue.update(mock_process)
        self.assertEqual(mock_process.call_count, 2)
        mock_process.assert_any_call("item1")
        mock_process.assert_any_call("item2")
        mock_process.reset_mock()
        
        time.sleep(0.1)
        self.queue.update(mock_process)
        mock_process.assert_called_once_with("item3")

if __name__ == '__main__':
    unittest.main()