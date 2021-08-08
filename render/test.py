from ga1axy import PyTrainer
import numpy as np

trainer = PyTrainer(
    batch_size=10,
    output_width=256,
    output_height=256,
    input_images=10,
    input_width=64,
    input_height=64,
    input_points=128,
)

print("Yeah!")
