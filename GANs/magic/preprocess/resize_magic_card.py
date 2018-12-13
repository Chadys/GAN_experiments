import cv2
import glob

max_h, max_w = 0, 0
for imagePath in glob.glob('../../../images/magic/*.jpg'):
    image = cv2.imread(imagePath)
    h, w = len(image), len(image[0])
    max_h, max_w = max(max_h, h), max(max_w, w)

max_h = max_w = max(max_h, max_w)

for imagePath in glob.glob('../../../images/magic/*.jpg'):
    image = cv2.imread(imagePath)
    h, w = len(image), len(image[0])
    left, top = max_w-w, max_h-h
    right, bottom = left//2, top//2
    left, top = left-right, top-bottom
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    cv2.imwrite(imagePath, new_im)

