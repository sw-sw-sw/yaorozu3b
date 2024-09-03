import tkinter as tk
from tkinter import ttk
import json
import os
from config_manager import ConfigManager

def run_parameter_control_ui(shared_memory, queues, running):
    root = tk.Tk()
    ui_to_tensorflow_queue = queues['ui_to_tensorflow']

    def update_callback(param_name, value):
        if param_name in shared_memory:
            shared_memory[param_name].value = value
        ui_to_tensorflow_queue.put((param_name, value))

    config_manager = ConfigManager()
    ui = ParameterControlUI(root, update_callback, config_manager)

    # 初期値の設定
    initial_values = {}
    for param_name in ui.parameters:
        if param_name.upper() in shared_memory:
            initial_values[param_name] = shared_memory[param_name.upper()].value
        else:
            initial_values[param_name] = config_manager.get_trait_value(param_name.upper())

    ui.set_initial_values(initial_values)

    def on_closing():
        running.value = False
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    def check_running():
        if running.value:
            root.after(100, check_running)
        else:
            root.quit()

    root.after(100, check_running)
    root.mainloop()

class ParameterControlUI:
    def __init__(self, root, update_callback, config_manager):
        self.root = root
        self.root.title("Simulation Parameter Control")
        self.update_callback = update_callback
        self.config_manager = config_manager
        self.sliders = {}
        self.value_labels = {}
        self.parameters = [
            'separation_distance', 'separation_weight',
            'cohesion_distance', 'cohesion_weight',
            'max_force', 'center_attraction_weight',
            'confinement_weight', 'rotation_strength',
            'escape_distance', 'escape_weight',
            'chase_distance', 'chase_weight'
        ]
        self.create_sliders()
        self.root.bind('<s>', self.save_settings)
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.grid(row=len(self.parameters), column=0, columnspan=3, pady=10)

    def create_sliders(self):
        for i, param_name in enumerate(self.parameters):
            label = " ".join(word.capitalize() for word in param_name.split('_'))
            ttk.Label(self.root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='e')
            
            min_val, max_val = self.config_manager.get_trait_range(param_name.upper())
            slider = ttk.Scale(self.root, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=i, column=1, padx=10, pady=5)
            slider.bind("<ButtonRelease-1>", lambda event, param=param_name: self.update_parameter(param, event.widget.get()))
            slider.bind("<B1-Motion>", lambda event, param=param_name: self.update_value_label(param, event.widget.get()))
            self.sliders[param_name] = slider

            value_label = ttk.Label(self.root, text="0.00")
            value_label.grid(row=i, column=2, padx=10, pady=5)
            self.value_labels[param_name] = value_label

    def update_parameter(self, param_name, value):
        self.update_callback(param_name, value)
        self.update_value_label(param_name, value)

    def update_value_label(self, param_name, value):
        self.value_labels[param_name].config(text=f"{value:.2f}")

    def set_initial_values(self, values):
        for param_name, value in values.items():
            if param_name in self.sliders:
                self.sliders[param_name].set(value)
                self.update_value_label(param_name, value)
                
    def save_settings(self, event=None):
        settings = {param: slider.get() for param, slider in self.sliders.items()}
        filename = 'simulation_settings.json'
        with open(filename, 'w') as f:
            json.dump(settings, f, indent=4)
        self.status_label.config(text=f"Settings saved to {filename}")
        self.root.after(3000, lambda: self.status_label.config(text=""))

    def load_settings(self):
        filename = 'simulation_settings.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                settings = json.load(f)
            self.set_initial_values(settings)
            for param, value in settings.items():
                self.update_callback(param, value)
            self.status_label.config(text=f"Settings loaded from {filename}")
            self.root.after(3000, lambda: self.status_label.config(text=""))