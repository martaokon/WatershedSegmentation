import cv2
import numpy as np

class watershedSegmentation:
    def __init__(self, imgs):
        self.imgs = imgs
        self.imgs_final = []
        self.imgs_markers = []
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

            # Background area determination - with dilation, to segment only background area
            background = cv2.dilate(opening, self.kernel)

            # Foreground area determination - with erosion, to segment only foreground area
            foreground = cv2.morphologyEx(thresh,cv2.MORPH_ERODE, self.kernel)

            # Finding unknown region with background and foreground subtraction
            foreground = np.uint8(foreground)
            unknown = cv2.subtract(background,foreground)

            # Marker labelling
            # connectedComponents() marks foreground with positive values and background with 0
            ret, markers = cv2.connectedComponents(foreground)

            # Because background is now labelled with 0, which is unknown region label 
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1

            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            # Apply watershed with Meyer's algorithm (already implemented function in openCV)
            markers = cv2.watershed(img,markers)

            # Apply watershed lines to image
            img[markers == -1] = [0,255,0]

            # Remember watershed markers
            self.imgs_markers.append(markers)
            # Remember segmented by watershed algorithm image
            self.imgs_final.append(img)

        return self.imgs_final, self.imgs_markers