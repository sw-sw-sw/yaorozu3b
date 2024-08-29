class TensorFlowSimulation:
    def __init__(self, queues, shared_memory):
        # 既存のコード...
        self.ui_to_tensorflow_queue = queues['ui_to_tensorflow']
        self.shared_memory = shared_memory

    def update_parameters(self):
        while not self.ui_to_tensorflow_queue.empty():
            param_name, value = self.ui_to_tensorflow_queue.get()
            setattr(self, param_name, tf.constant(value, dtype=tf.float32))

    def run(self):
        while self.running.value:
            self.update_parameters()
            # 既存のシミュレーションコード...