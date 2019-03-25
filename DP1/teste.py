import numpy as np
from scipy.spatial import distance
import cv2 

def select_same_color(pixel):
    height, width, _ = img.shape
    print "width: {}, height: {}".format(width, height)
    for x in range (0,height):
        for y in range (0,width):
            dist = distance.euclidean(img[x, y], pixel)
            if (dist < 50):
                img[x, y] = [0, 0, 255]
    cv2.imshow('image2',img)            
                

img = cv2.imread('croc.jpg', cv2.IMREAD_COLOR) 
cv2.imshow('image',img)

px = [0, 0, 0]
select_same_color(px)

cv2.waitKey(0)
cv2.destroyAllWindows()