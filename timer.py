import time

class Timer:
    def __init__(self, name: str, fps_update_interval = 0.01):
        self.name = name
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_update_interval = fps_update_interval
        self.last_time = time.time()
        self.time_value = 0
        self.b_print = True
        self.total_time = 0
    
    def start(self):
        self.start_time = time.time()
        
    
    def _calculate_fps(self):
        return self.frame_count / (self.current_time - self.last_time) if (self.current_time - self.last_time) > 0 else 0

    def sleep_time(self, other_time_value):
        sleep_duration = max(other_time_value - self.time_value, 0)
        # print(f"Sleeping for {sleep_duration:.6f} seconds")
        time.sleep(sleep_duration)
        
    def calculate_time(self):
        self.time_value = time.time() - self.start_time
        return self.time_value
    
    def print_lap_time(self,interval_time):
        self.current_time = time.time() 
        if self.current_time - self.last_time >= interval_time:
            lap_time = self.calculate_time() * 1000
            print(self.name + " Lap time(ms):", f"{lap_time:4.2f}")
            self.last_time = self.current_time
        return
    
    def interval_timer(self,interval_time):
        self.current_time = time.time() 
        if self.current_time - self.last_time >= interval_time:
            self.last_time = self.current_time
            return True
        return False
    
    def print_lap_fps(self,interval_time):
        self.current_time = time.time() 
        if self.current_time - self.last_time >= interval_time:
            lap_fps = 1 / self.calculate_time() 
            print(self.name + " Lap FPS:", f"{lap_fps:4.2f}")
            self.last_time = self.current_time
        return
    
    def print_fps(self,interval_time):
        self.frame_count += 1
        self.current_time = time.time() 
        if self.current_time - self.last_time >= interval_time:
            # calculate fps
            fps = self._calculate_fps()
            print(self.name + " FPS:", f"{fps:4.2f}")
            self.last_time = self.current_time
            self.frame_count = 0            
        return

    def print_average_time(self,interval_time):
        lap_time = time.time() - self.start_time
        self.total_time += lap_time
        self.frame_count += 1
        self.current_time = time.time() 
        if self.current_time - self.last_time >= interval_time:
            # calculate fps
            average_time = self.total_time / self.frame_count
            print(self.name + " :", f"{average_time * 1000:4.2f} ms")
            self.last_time = self.current_time
            self.total_time = 0
            self.frame_count = 0            
        return