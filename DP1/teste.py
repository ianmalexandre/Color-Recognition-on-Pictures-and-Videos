from scipy.misc import imread, imsave, imresize
import cv2

image = cv2.imread('images/grey.jpeg', -1)
if(len(image.shape)<3):
      print 'gray'
elif len(image.shape)==3:
      print 'Color(RGB)'
else:
      print 'others'