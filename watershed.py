import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
from scipy import ndimage

class watershedSegmentation:
    def __init__(self, imgs):
        self.imgs = imgs
        self.imgs_final = []
        self.kernel = np.ones((5,5),np.uint8)
        return 

    # MARKER-CONTROLLED WATERSHED SEGMENTATION ALGORITHM
    def segmentize(self):
        for i, img in enumerate(self.imgs):
            # Converting to grayscale
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

            # Noise removal - opening morphological transformation gives in this case better results then closing.
            opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, self.kernel)
            # closing = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, self.kernel)

            # Background area determination
            background = cv2.dilate(opening, self.kernel)

            # Foreground area determination - with euclidean distance (DIST_L2)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            ret, foreground = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)

            # Finding unknown region
            foreground = np.uint8(foreground)
            unknown = cv2.subtract(background,foreground)

            # Marker labelling
            ret, markers = cv2.connectedComponents(foreground)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1

            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            # Apply watershed with Meyer's algorithm
            markers = cv2.watershed(img,markers)
            img[markers == -1] = [0,255,0]

            # Remember segmented by watershed algorithm image
            self.imgs_final.append(img)

        return self.imgs_final