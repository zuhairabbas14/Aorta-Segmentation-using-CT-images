#!/usr/bin/env python
# coding: utf-8

# In[4]:


import math
from cv2 import erode
from sklearn.cluster import DBSCAN as Dbscan


def get_contour(imgg):

    imgg = cv2.medianBlur(imgg, 7)
    feature_array = np.empty((512,512,3))

    for x in range(512):
        for y in range(512):
            feature_array[x][y] = (x, y, (imgg[x][y] - 1000) * 5 if 1000 < imgg[x][y] < 1100 else 0)

    db = Dbscan(eps = 15, min_samples=15)
    predicted = db.fit_predict(feature_array.flatten().reshape(-1, 3)).reshape(512,512)

    img = predicted.astype('uint8')
    kernel = np.ones((2,2),dtype=np.uint8) # this must be tuned 
    img=erode(img,kernel)
    img = cv2.medianBlur(img, 5)

    img2 = img.copy()
    contours,hierarchy = cv2.findContours(img,cv2.RETR_TREE,1)
    cir = 0

    for c in contours:

        perimeter = cv2.arcLength(c, True)
        area = cv2.contourArea(c)
        # GET center of contours
        M = cv2.moments(c)  

        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        except ZeroDivisionError:
            cX = int(M["m10"])
            cY = int(M["m01"])

        if perimeter == 0 or (area > 2000 or area < 500) or (cX < 220 or cX > 320) or (cY < 250 or cY > 300):
            continue

        else:

            print('center', cX, cY)

            (x,y),radius = cv2.minEnclosingCircle(c)
            center = (int(x),int(y))
            radius = int(radius)
            circularity = 4*math.pi*(area/(perimeter*perimeter))

            if 0.2 < circularity < 1.2:
                cir +=1
                cv2.drawContours(imgg, [c], -1, (255, 255, 0), 3)
                cv2.drawContours(img, [c], -1, (255, 255, 0), 3)

            # print("<> Total circles " +  str(cir) + "\n\n")
    #         print("\nPerimeter: " + str(perimeter))
    #         print("Area: " + str(area))
    #         print("Diameter: " + str(radius*2))

    # fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    # ax = axes.flatten()
    # ax[0].imshow(imgg, cmap='gray')
    # ax[1].imshow(img, cmap='gray')


    def store_evolution_in(lst):
        """Returns a callback function to store the evolution of the level sets in
        the given list.
        """

        def _store(x):
            lst.append(np.copy(x))

        return _store

    # Morphological ACWE
    image = img_as_float(img)

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    # List with intermediate results for plotting the evolution
    evolution = []
    callback = store_evolution_in(evolution)
    ls = morphological_chan_vese(image, 37, init_level_set=init_ls, smoothing=3,
                                 iter_callback=callback)

    fig, axes = plt.subplots(1, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].contour(ls, [0.5], colors='r')
    ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

    ax[1].imshow(ls, cmap="gray")
    ax[1].set_axis_off()
    contour = ax[1].contour(evolution[2], [0.5], colors='g')
    contour.collections[0].set_label("Iteration 2")
    contour = ax[1].contour(evolution[7], [0.5], colors='y')
    contour.collections[0].set_label("Iteration 7")
    contour = ax[1].contour(evolution[-1], [0.5], colors='r')
    contour.collections[0].set_label("Iteration 37")

    ax[1].legend(loc="upper right")
    title = "Morphological ACWE evolution"
    ax[1].set_title(title, fontsize=12)

    plt.show()

