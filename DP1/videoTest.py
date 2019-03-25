import numpy as np
import cv2

point = []
px = []

def fusion_img2mask(mask):
	
	#reference: https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
	# I want to put logo on top-left corner, So I create a ROI
	new_img = img.astype(np.uint8)	
	rows,cols,channels = mask.shape
	roi = new_img[0:rows, 0:cols]	
	
	# Now create a mask of logo and create its inverse mask also
	img2gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
	ret, mask2 = cv2.threshold(img2gray, 127, 255, cv2.THRESH_BINARY)
	mask_inv = cv2.bitwise_not(mask2)

	# Now black-out the area of logo in ROI
	img1_bg = cv2.bitwise_and(roi,roi,mask = mask2)	

	# Take only region of logo from logo image.
	img2_fg = cv2.bitwise_and(mask,mask,mask = mask_inv)	
	img2_fg = img2_fg.astype(np.uint8)
	img1_bg = img1_bg.astype(np.uint8)	
	
	# Put logo in ROI and modify the main image
	dst = cv2.add(img1_bg,img2_fg)
	#img[0:rows, 0:cols] = dst

	cv2.imshow('final', dst)


def select_same_color_2(pixel):
	width, height, depth = img.shape
	maskPixel = np.zeros(((width, height, depth)))
	maskPixel[:,:] = pixel
	#b = (img[:,:,0] - mask1[:,:,0])
	#g = (img[:,:,1] - mask1[:,:,1])
	#r = (img[:,:,2] - mask1[:,:,2])
	res = np.sqrt(((img[:,:,0] - maskPixel[:,:,0])**2)+((img[:,:,1] - maskPixel[:,:,1])**2)+((img[:,:,2] - maskPixel[:,:,2])**2))
	res = np.where(res < 13, 0, 255)
	mask = np.zeros(((width, height,3))) 
	mask[:,:,0] = res
	mask[:,:,1] = res
	mask[:,:,2] = 255
	mask = mask.astype(np.uint8)
	cv2.imshow('mask', mask)
	fusion_img2mask(mask)

    


def get_mouse_clicks(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print "Point clicked: ".format(x, y)
        point = [x,y]
        px = img[point[1], point[0]]
        print px
        select_same_color_2(px)
        

cap = cv2.VideoCapture('video.avi')

while(cap.isOpened()):
	ret, img = cap.read()

	cv2.imshow('frame',img)	
	cv2.setMouseCallback("frame", get_mouse_clicks)

			
	if cv2.waitKey(1) & 0xFF == ord('q'):
	    break


cap.release()
cv2.destroyAllWindows()

