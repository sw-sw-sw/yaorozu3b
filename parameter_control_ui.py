import tkinter as tk
from tkinter import ttk

class ParameterControlUI:
    def __init__(self, root, update_callback):
        self.root = root
        self.root.title("Simulation Parameter Control")
        self.update_callback = update_callback
        self.sliders = {}
        self.create_sliders()

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
            ttk.Label(self.root, text=label).grid(row=i, column=0, padx=10, pady=5)
            slider = ttk.Scale(self.root, from_=min_val, to=max_val, orient=tk.HORIZONTAL, length=200)
            slider.grid(row=i, column=1, padx=10, pady=5)
            slider.bind("<ButtonRelease-1>", lambda event, param=param_name: self.update_parameter(param, event.widget.get()))
            self.sliders[param_name] = slider

    def update_parameter(self, param_name, value):
        self.update_callback(param_name, value)

    def set_initial_values(self, values):
        for param_name, value in values.items():
            if param_name in self.sliders:
                self.sliders[param_name].set(value)