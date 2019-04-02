import numpy as np
import cv2

#Declaration of global variables:
px = []
videoOrPhoto = 0

def fusion_img2mask(mask, flag, new_img):
	
	#reference: https://docs.opencv.org/3.2.0/d0/d86/tutorial_py_image_arithmetics.html
	# I want to put logo on top-left corner, So I create a ROI
		
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

def forming_mask(res, width, height):
	res = np.where(res < 13, 0, 255)
	mask = np.zeros(((width, height,3))) 
	mask[:,:,0] = res
	mask[:,:,1] = res
	mask[:,:,2] = 255
	mask = mask.astype(np.uint8)
	return mask

def select_same_color_1(pixel):
	flag = 2
	width, height= img.shape
	maskPixel = np.full((width, height), pixel)
	res = img[:,:] - maskPixel[:,:]
	mask = forming_mask(res, width, height)
	new_img = np.zeros(((width, height,3)))
	new_img[:,:,0] = img.astype(np.uint8)
	new_img[:,:,1] = img.astype(np.uint8)
	new_img[:,:,2] = img.astype(np.uint8)
	fusion_img2mask(mask, flag, new_img)
	



def select_same_color_2(pixel):
	flag = 1
	width, height, depth = img.shape
	maskPixel = np.zeros(((width, height, depth)))
	maskPixel[:,:] = pixel
	#b = (img[:,:,0] - mask1[:,:,0])
	#g = (img[:,:,1] - mask1[:,:,1])
	#r = (img[:,:,2] - mask1[:,:,2])
	res = np.sqrt(((img[:,:,0] - maskPixel[:,:,0])**2)+((img[:,:,1] - maskPixel[:,:,1])**2)+((img[:,:,2] - maskPixel[:,:,2])**2))
	mask = forming_mask(res, width, height)
	new_img = img.astype(np.uint8)
	fusion_img2mask(mask, flag, new_img)



def get_mouse_clicks(event, x, y, flags, params):
	global px
	global videoOrPhoto
	if event == cv2.EVENT_LBUTTONDOWN:
		print "Point clicked: {}, {}".format(x, y)
		px = img[y, x]
		if videoOrPhoto == 2:
			if RGBorGrey == 1:
				select_same_color_2(px)
				print "B = {}, G = {}, R = {}". format(px[0], px[1], px[2])
			else:
				select_same_color_1(px)
				print "L = ", px	

if __name__ == '__main__':

	name = raw_input("Name of your file: ")

	while True:
		videoOrPhoto = input("1 for Video 2 for Image: ")
		if videoOrPhoto == 1:
			break
		elif videoOrPhoto == 2:
			break	

	if videoOrPhoto == 1:
		while True:
			cameraOrVideo = input("1 for Video 2 for Camera: ")
			if cameraOrVideo == 1:
				cap = cv2.VideoCapture('videos/' + name)
				break
			elif cameraOrVideo == 2:
				cap = cv2.VideoCapture(0)
				break	
	
		cv2.namedWindow("frame")

		cv2.setMouseCallback("frame", get_mouse_clicks)


		while(cap.isOpened()):

			ret, img = cap.read()
			if ret:
				cv2.imshow('frame', img)
				if len(px) > 0:
					select_same_color_2(px)
			else:
				break

			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		cap.release()
		cv2.destroyAllWindows()
	elif videoOrPhoto == 2:
		
		while True:
			RGBorGrey = input("1 for RGB 2 for Greyscaled: ")
			if RGBorGrey == 1:
				img = cv2.imread('images/' + name, 1)
				break
			elif RGBorGrey == 2:
				img = cv2.imread('images/' + name, 0)
				break	
	
		cv2.namedWindow("image")

		#img = cv2.imread('rainbow.jpg',cv2.IMREAD_GRAYSCALE)

		cv2.imshow('image',img)

		#Reference: https://docs.opencv.org/2.4/modules/highgui/doc/user_interface.html?highlight=setmousecallback

		cv2.setMouseCallback("image", get_mouse_clicks)

		cv2.waitKey(0)
		cv2.destroyAllWindows()

