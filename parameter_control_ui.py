import tkinter as tk
from tkinter import ttk
import json
import os

class ParameterControlUI:
    def __init__(self, root, update_callback):
        self.root = root
        self.root.title("Simulation Parameter Control")
        self.update_callback = update_callback
        self.sliders = {}
        self.value_labels = {}  # 新しく追加：値を表示するラベルの辞書
        self.create_sliders()
        self.root.bind('<s>', self.save_settings)  # Add key binding for 's'

    def create_sliders(self):
        parameters = [
            ("Separation Distance", 'separation_distance', 0, 100),
            ("Separation Weight", 'separation_weight', 0, 50),
            ("Cohesion Distance", 'cohesion_distance', 0, 500),
            ("Cohesion Weight", 'cohesion_weight', 0, 50),
            ("Max Force", 'max_force', 0, 500),
            ("Center Attraction Weight", 'center_attraction_weight', 0, 200),
            ("Confinement Weight", 'confinement_weight', 0, 200),
            ("Rotation Strength", 'rotation_strength', 0, 600),
        ]

        for i, (label, param_name, min_val, max_val) in enumerate(parameters):
            ttk.Label(self.root, text=label).grid(row=i, column=0, padx=10, pady=5, sticky='e')
            
            slider = ttk.Scale(self.root, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=i, column=1, padx=10, pady=5)
            slider.bind("<ButtonRelease-1>", lambda event, param=param_name: self.update_parameter(param, event.widget.get()))
            slider.bind("<B1-Motion>", lambda event, param=param_name: self.update_value_label(param, event.widget.get()))
            self.sliders[param_name] = slider

            # 値を表示するラベルを追加
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
        self.root.after(3000, lambda: self.status_label.config(text=""))  # Clear message after 3 seconds

    def load_settings(self):
        filename = 'simulation_settings.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                settings = json.load(f)
            self.set_initial_values(settings)
            for param, value in settings.items():
                self.update_callback(param, value)
            self.status_label.config(text=f"Settings loaded from {filename}")
            self.root.after(3000, lambda: self.status_label.config(text=""))  # Clear message after 3 seconds