def canny(image):
    borders = np.empty_like(image)
    max_color = image.max()
    min_color = image.min()
    for i, slice_ in enumerate(image):
        slice_ -= min_color
        slice_ = (slice_ / max_color * 256).astype('uint8')
        cvt = cv2.cvtColor(slice_, cv2.COLOR_GRAY2RGB)
        border = cv2.Canny(cvt, 100, 200)
        borders[i] = border
    return borders
