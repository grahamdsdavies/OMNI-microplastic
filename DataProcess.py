
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

img = cv2.imread("type3.jpg")
# img = cv2.resize(img, (768,576))
#initialising indexes to store inputs from clicks
pts = [(0,0),(0,0),(0,0),(0,0)]
pointIndex = 0
AR = (1280, 900)
oppts = np.float32([[0,0],[AR[1],0],[0,AR[0]],[AR[1],AR[0]]])

#function to select four points on a image to capture desired region
def draw_circle(event,x,y,flags,param):
	global img
	global pointIndex
	global pts

	if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img,(x,y),5,(0,255,0),-1)
                pts[pointIndex] = (x,y)
                #print(pointIndex)
                pointIndex = pointIndex + 1
               
def show_window():                       
        while True:
                #print(pts,pointIndex-1)
                cv2.imshow('img', img)
                
                if(pointIndex == 4):
                        break
                
                if (cv2.waitKey(20) & 0xFF == 27) :
                        break

def get_persp(image,pts):
        ippts = np.float32(pts)
        Map = cv2.getPerspectiveTransform(ippts,oppts)
        warped = cv2.warpPerspective(image, Map, (AR[1], AR[0]))
        return warped

cv2.namedWindow('img')
cv2.setMouseCallback('img',draw_circle)
print('Top left, Top right, Bottom Left, Bottom Right')

show_window()

while True:
        #_, frame = cap.read()
        warped = get_persp(img, pts)
        # cv2.imshow("output",warped)

        #save output file in same path
        # cv2.imwrite("output.jpg",warped)
        # key = cv2.waitKey(10) & 0xFF
        if 1:
                break
cv2.destroyAllWindows()


# Read image
# im = cv2.imread("type3.jpg", cv2.IMREAD_GRAYSCALE)
im = warped

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 100

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = True
# params.minConvexity = 0.87
params.minConvexity = 0.1

    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)
print(len(keypoints))
for keyPoint in keypoints:
    x = keyPoint.pt[0]
    y = keyPoint.pt[1]
    s = keyPoint.size
print(s)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# plt.hist(s, density=True, bins=30)  # density=False would make counts
# plt.show()

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)