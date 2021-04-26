import cv2
import numpy as np
from matplotlib import pyplot as plt

# image = cv2.imread('type4.jpg')
image = cv2.imread('sorted.jpg')
RGB_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
thresh = cv2.threshold(blur,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
cnts = cv2.findContours(opening, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

ROI_number = 0
for c in cnts:
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x,y,w,h = cv2.boundingRect(approx)
    if len(approx) == 4 and (area > 1000000) and (area < 8000000):
        ROI = image[y:y+h, x:x+w]
        cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
        ROI_number += 1

# cv2.imshow('thresh', thresh)
# cv2.imshow('opening', opening)
# cv2.waitKey()

# im = cv2.imread("type3.jpg", cv2.IMREAD_GRAYSCALE)
im = ROI
RGB_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 70

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
# params.minConvexity = 0.87
params.minConvexity = 0.1

    
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)
print(keypoints)
s=[]
c=[]
h=[]
xl=[]
yl=[]
for keyPoint in keypoints:
    x = int(keyPoint.pt[0])
    y = int(keyPoint.pt[1])
    xl.append(x)
    yl.append(y)
    # print(keyPoint)
    s.append(keyPoint.size) #diameter in pixels
    c.append(RGB_im[y,x]/256) #diameter in pixels
    h.append(1)
    print(image[x,y])
print(s)

# implot = plt.imshow(RGB_im)
# plt.plot(xl,yl,'o')
plt.show()

# cv2.imshow(image)
# plt.matshow(RGB_image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.bar(s, h, color=c, width=7)
plt.xticks([])
plt.yticks([])
plt.xlabel('Microplastic Size →')
plt.show()


# plt.xlabel('Microplastic Size')
# plt.ylabel('Frequncy')
test=plt.hist(s, density=True, bins=4)  # density=False would make counts
mylabels = ["V Small", "Small", "Medium", "Large"]

plt.pie(test[0], labels = mylabels, counterclock=False, startangle=90, normalize=True)
plt.show() 
print(test[0])
# plt.show()

plt.hist(s, density=True, bins=20)  # density=False would make counts
plt.xlabel('Microplastic Size')
plt.ylabel('Number')
plt.show() 



# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)