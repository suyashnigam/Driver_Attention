import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
from skimage import io

Blur_size = 3
PostConstant = 0.97
thresholdGradient = 50.0
prepro = True
postpro =True
new_width = 20
np.set_printoptions(threshold='nan')

def computeGradient(img,dir):
	assert (dir == 'x' or dir == 'y'), "Direction must be x or y" 
	if dir=='x':
		grad = cv2.Sobel(img,cv2.CV_32F,dx=1,dy=0,ksize=3)
		
	else:
		grad = cv2.Sobel(img,cv2.CV_32F,dx=0,dy=1,ksize=3)
		
	return grad

def preprocess(img,new_width=20):
	img = cv2.GaussianBlur(img,(Blur_size,Blur_size),0)
	h, w = img.shape
	img = cv2.resize(img, (new_width,new_width), interpolation = cv2.INTER_LINEAR)
	return img

def get_score(img,x,y,gx,gy,weight, score):
	width = img.shape[0]
	height = img.shape[1]
	# score = 0
	for xi in range(0,width):
		for yi in range(0,height):
			if(xi==x and yi==y):
				score[yi][xi]+=0
			else:
				dx = xi - x
				dy = yi - y
				magnitude = math.sqrt((dx*dx)+(dy*dy))
				dx = dx/magnitude
				dy = dy/magnitude

				dotProd = dx*gx + dy*gy
				# dotProd = max(0,dotProd)

				dotProd = dotProd*dotProd*weight
				score[yi][xi]  += dotProd
	return score

def get_weight(img):
	img = cv2.bitwise_not(img)
	cv2.imwrite("Inverted.jpg",img)
	return img


def findCenter(img):

	if (prepro==True):
		img = preprocess(img,new_width)
	img_grad_X = computeGradient(img,'x') 
	img_grad_Y = computeGradient(img,'y')

	
	score = np.zeros(img.shape)

	#TODO check it
	weight = get_weight(img)
	# weight = np.eye(new_width,new_width)

	for y in range(0,img.shape[0]):
		for x in range(0,img.shape[1]):
			gx = img_grad_X[y][x]
			gy = img_grad_Y[y][x]

			magnitude = math.sqrt((gx*gx)+(gy*gy))
			

			if magnitude>thresholdGradient:
				gx = gx/magnitude
				gy = gy/magnitude
			else:
				gx = 0.0
				gy = 0.0

			# score[i][j] = get_score(img,i,j,gx,gy,weight[i][j])
			score = get_score(img,x,y,gx,gy,weight[y][x], score)

	# print score		
	maxVal = np.amax(score)
	if (postpro==True):
		thresh = PostConstant*maxVal
	# print thresh
	super_threshold_indices = score > thresh
	score[super_threshold_indices] = 0
	score_new = normalize(score)
	
	return np.unravel_index(np.argmax(score),score.shape)

def normalize(score):
	maxi = np.amax(score)
	mini = np.amin(score)
	score -= mini
	score /= maxi 
	score *=255
	return score

def relative(x,y,img):
	x = int((img.shape[0]/new_width)*x)
	y = int((img.shape[1]/new_width)*y)
	return x,y

def getGaze(image):
	
	print (image.shape)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print (img.shape)

	x,y = findCenter(img)
	
	

	
	x_new,y_new = relative(x,y,img)
	cv2.circle(img, (x_new,y_new), 1, 255, -1)

	print ("Gaze centred at {} and {}".format(x_new,y_new))

	return x_new,y_new

	
if __name__ == '__main__':
	main()