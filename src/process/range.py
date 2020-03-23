def window_image(image, center, width):
    left, right = center - width, center + width
    copy = image.copy()
    copy[copy < left] = left
    copy[copy > right] = right
    return copy