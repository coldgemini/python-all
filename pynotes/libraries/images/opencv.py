while True:
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

import cv2
import numpy as np
import Image

img = cv2.imread('img7.jpg')
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
a = np.asarray(gray_image)

dst = np.zeros(shape=(5, 2))

b = cv2.normalize(a, dst, 0, 255, cv2.NORM_L1)
norm_image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

im = Image.fromarray(b)

im.save("img50.jpg")

cv2.waitKey(0)
cv2.destroyAllWindows()

import numpy as np
import cv2 as cv

im = cv.imread('tests.jpg')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
im2, contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

cv.drawContours(img, contours, -1, (0, 255, 0), 3)
cv.drawContours(img, contours, 3, (0, 255, 0), 3)
cnt = contours[4]
cv.drawContours(img, [cnt], 0, (0, 255, 0), 3)

# for name in files:
fidx = 0
ifStop = False
while not ifStop:
    name = files[fidx]
    cv2.imshow(' ', binaryImg)
    while True:
        key = cv2.waitKey(0)
        if key == ord('f'):
            fidx += 1
            break
        if key == ord('b'):
            fidx -= 1
            break
        if key == ord('q'):
            ifStop = True
            break

image_blur = cv2.medianBlur(image_gray, 5)
img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_NEAREST)

# add mask color
msk[:, :, 0] = 0
msk[:, :, 1] = 0
msk = msk // 6
print(np.unique(msk))
masked_img = img + msk

cv2.ellipse(mask, (100, 100), (100, 20), 45, startAngle=0, endAngle=360, color=255, thickness=-1)

cv2.line(img, pt1, pt2, [255, 0, 0], thickness=3)

# convert image to grayscale image
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert the grayscale image to binary image
ret,thresh = cv2.threshold(gray_image,127,255,0)

# calculate moments of binary image
M = cv2.moments(thresh)

# calculate x,y coordinate of center
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# put text and highlight the center
cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# display the image
cv2.imshow("Image", img)
cv2.waitKey(0)