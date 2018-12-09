import cv2
import glob
from collections import Counter
from operator import itemgetter

anomalies = []
min_h, min_w = 10000, 10000
for imagePath in glob.glob('../images/magic/*.jpg'):
    image = cv2.imread(imagePath)
    h, w = len(image), len(image[0])
    min_h, min_w = min_h if min_h < h else h, min_w if min_w < w else w

    if [h, w] not in [[445, 312], [310, 217], [445, 315], [442, 312], [285, 200], [370, 265]]:
        anomalies.append((h, w))
        # print(imagePath)
        # print(f'{h}, {w}')
        continue
    black_threshold = 55
    # manage all existing card format
    if [h, w] == [310, 217]:
        image = image[32:172, 14:204]
    elif [h, w] == [445, 315]:
        image = image[47:246, 20:294]
    elif [h, w] == [442, 312]:
        image = image[36:239, 31:283]
    elif [h, w] == [285, 200]:
        image = image[27:158, 10:191]
    elif [h, w] == [370, 265]:
        image = image[43:204, 22:243]
    # Check if a precise pixel is black
    elif all(image[403, 270] < black_threshold)\
            or any(all(x < black_threshold) for x in image[383, 11:16]):
        image = image[48:48+198, 21:21+270]
    elif all(image[405, 270] < black_threshold+20):
        image = image[51:51+196, 24:24+264]
    else:
        image = image[36:36+205, 30:30+252]

    imin = 0
    limit = h/3
    for i, line in enumerate(image):
        if i > limit:
            break
        maxval = [max(line, key=itemgetter(j))[j] for j in range(0,3)]
        minval = [min(line, key=itemgetter(j))[j] for j in range(0,3)]
        if all(val < black_threshold for val in [m1-m2 for m1,m2 in zip(maxval,minval)]): #detect black line
            imin = i
    image = image[imin:]
    imagePath = imagePath.replace("magic", "magic_img")
    cv2.imwrite(imagePath, image)

print(Counter(anomalies))
