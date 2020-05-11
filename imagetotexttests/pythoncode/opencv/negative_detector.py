import numpy as np
from PIL import Image
import os

def get_image(image_path):
    """Get a numpy array of an image so that one can access values[x][y]."""
    image = Image.open(image_path, 'r')
    width, height = image.size
    pixel_values = list(image.getdata())
    if image.mode == 'RGB':
        channels = 3
    elif image.mode == 'L':
        channels = 1
    else:
        print("Unknown mode: %s" % image.mode)
        return None
    pixel_values = np.array(pixel_values).reshape((width, height, channels))
    return pixel_values

def get_white_percentage(image_name):
    numpy_array = get_image("Sample Resources/" + image_name)
    total_pixels = numpy_array.size
    num_of_white = np.count_nonzero(numpy_array == [255,255, 255])
    num_of_black = np.count_nonzero(numpy_array == [1,1, 1])
    return num_of_white / total_pixels

for filename in os.listdir("../Sample Resources/"):
    if filename.endswith(".jpg"):
        white_percentage = get_white_percentage(filename)
        print(filename)
        print(white_percentage)