import cv2
import os
import matplotlib.pyplot as plt

def loadImgs(dir = 'berlin'):
    imgs = []

    for fileName in os.listdir(dir):
        img = cv2.imread(dir + '/' + fileName)
        imgs.append(img)
    return imgs

def saveImgs(imgs, dir = 'berlin'):
    idx = 0
    for fileName in os.listdir(dir):
        cv2.imwrite(dir + '/' + fileName.split('.')[0] + "_result.png", imgs[idx])
        idx = idx + 1

def showImgs(imgs):
    idx = 1
    for img in imgs:
        plt.subplot(2,2,idx),plt.imshow(img,cmap = 'gray')
        idx = idx+1
    plt.show()