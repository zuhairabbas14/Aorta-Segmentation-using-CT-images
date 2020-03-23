from pathlib import Path
import numpy as np
import cv2 as cv
from pydicom import dcmread
import napari

def show_image(*images):
    viewer = napari.Viewer()
    for index, image in enumerate(images):
        im, sc = image[0], image[1]
        viewer.add_image(im, scale=(sc[0], sc[1], sc[1]), name = "image"+str(index))

def read_serie(path):
    directory = Path(path)
    slices = [dcmread(str(filename)) for filename in directory.iterdir()]
    data = [(slice_.SliceLocation, slice_.pixel_array + int(slice_.RescaleIntercept)) for slice_ in slices]
    data = sorted(data)
    image = np.array([d[1] for d in data])
    scale = slices[0].SliceThickness, slices[0].PixelSpacing[0]
    return image, scale
