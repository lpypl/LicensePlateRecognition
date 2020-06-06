import cv2
import numpy as np
import imageProcess as ip

fpath = r'/home/lpy/PythonProjects/LicensePlateRecognition/dataset/example00.bmp'
img = cv2.imread(fpath)

grayimg = ip.img2gray(img)
grayimg = cv2.equalizeHist(grayimg)
x_edges, y_edges = ip.edgeDetection(grayimg)
for pixel in x_edges + y_edges:
    img[pixel[0], pixel[1]] = np.array([0, 0, 255])

# binimg = ip.gray2binary(grayimg)
cv2.imshow('', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
