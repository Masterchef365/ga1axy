from ga1axy import PyTrainer, visualize_inputs
import numpy as np

batch_size = 10
output_width = 256
output_height = 256
input_images = 10
input_width = 64
input_height = 64
input_points = 128

trainer = PyTrainer(
    batch_size=batch_size,
    output_width=output_width,
    output_height=output_height,
    input_images=input_images,
    input_width=input_width,
    input_height=input_height,
    input_points=input_points,
)

rng = np.random.RandomState()

images = rng.rand(batch_size, input_images, input_height, input_width, 4)
images = (images * 255.0).astype('uint8')

points = rng.rand(batch_size, input_points, 4).astype('float32')

output = trainer.frame(points, images)

from PIL import Image
for img in output:
    img = Image.fromarray(img, 'RGB')
    img.show()

visualize_inputs(points, images)
