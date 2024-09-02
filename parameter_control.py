from parameter_control_ui import ParameterControlUI  
import tkinter as tk 


def run_parameter_control_ui(shared_memory, queues, running):
    root = tk.Tk()
    ui_to_tensorflow_queue = queues['ui_to_tensorflow']

    def update_callback(param_name, value):
        shared_memory[param_name].value = value
        ui_to_tensorflow_queue.put((param_name, value))

    ui = ParameterControlUI(root, update_callback)

    # 初期値の設定
    initial_values = {
        'separation_distance': shared_memory['separation_distance'].value,
        'separation_weight': shared_memory['separation_weight'].value,
        'cohesion_distance': shared_memory['cohesion_distance'].value,
        'cohesion_weight': shared_memory['cohesion_weight'].value,
        'max_force': shared_memory['max_force'].value,
        'center_attraction_weight': shared_memory['center_attraction_weight'].value,
        'confinement_weight': shared_memory['confinement_weight'].value,
        'rotation_strength': shared_memory['rotation_strength'].value
    }
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