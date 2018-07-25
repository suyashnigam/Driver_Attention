import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import time
from skimage import io
# from PIL import image

# def scale(img,fast_width):
# 	img = cv2.resize(img,(fast_width,fast_width))
# 	return img

## CONSTANTS
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
		
		# cv2.imwrite('gradx.jpg',grad)
		# grad = cv2.normalize(grad, grad, -127.0, 127.0, cv2.NORM_MINMAX, cv2.CV_8U)

	else:
		grad = cv2.Sobel(img,cv2.CV_32F,dx=0,dy=1,ksize=3)
		# print ("Gradient Y", grad)

		# cv2.imwrite('grady.jpg',grad)

		# grad = cv2.normalize(grad, grad, -127.0, 127.0, cv2.NORM_MINMAX, cv2.CV_8U)

	return grad

def preprocess(img,new_width=20):
	img = cv2.GaussianBlur(img,(Blur_size,Blur_size),0)
	h, w = img.shape
	# r = new_width / w
	# dim = (new_width, int(h * r))
	# img = cv2.resize(img, dim, interpolation = cv2.INTER_LINEAR)
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
	# cv2.imshow("score",score_new)
	# cv2.waitKey(0)
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
	# print ("Finding gaze centre")
	# img = cv2.imread('4.jpg',0)
	print (image.shape)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	print (img.shape)

	x,y = findCenter(img)
	
	

	# print "Prior to scaling:",x,y
	# img = cv2.resize(img, (new_width,new_width), interpolation = cv2.INTER_LINEAR)
	# cv2.imwrite("Orignal.jpg",img)
	x_new,y_new = relative(x,y,img)
	cv2.circle(img, (x_new,y_new), 1, 255, -1)

	print ("Gaze centred at {} and {}".format(x_new,y_new))

	return x_new,y_new

	# cv2.imwrite('4_gaze.jpg', img)
	# img = cv2.resize(img, (new_width,new_width), interpolation = cv2.INTER_LINEAR)
	# cv2.circle(img, (x,y), 1, 255, -1)
	# cv2.imwrite('unscaled.jpg',img)

	# return x,y

def GazeAngle(eye_centre,face_centre,gaze_centre,face_rad):
	origin = [0,0,0]
	X,Y,Z = face_centre[0],face_centre[1],face_rad 
	Y_shifted = Y-eye_centre[1]
	origin_shifted = [0,Y_shifted,0]

	X_gaze_shifted = X - gaze_centre[0]
	Y_gaze_shifted = Y - gaze_centre[1]
	Z_gaze_shifted = Z 

	X_eye_shifted = X - eye_centre[0]
	Y_eye_shifted = X - eye_centre[1]
	Z_eye_shifted = Z

	Gaze = [X_gaze_shifted,Y_gaze_shifted,Z_gaze_shifted]
	Eye = [X_eye_shifted,Y_eye_shifted,Z_eye_shifted]

	Gaze_Vector = np.zeros(3)
	Eye_Vector = np.zeros(3)

	Gaze_Vector[0] = Gaze[0] - origin_shifted[0]
	Gaze_Vector[1] = Gaze[1] - origin_shifted[1]
	Gaze_Vector[2] = Gaze[2] - origin_shifted[2]

	Eye_Vector[0] = Eye[0] - origin_shifted[0]
	Eye_Vector[1] = Eye[1] - origin_shifted[1]
	Eye_Vector[2] = Eye[2] - origin_shifted[2]


	Gaze_magnitude = np.sqrt(Gaze_Vector[0]*Gaze_Vector[0]+Gaze_Vector[1]*Gaze_Vector[1]+Gaze_Vector[2]*Gaze_Vector[2])
	Eye_magnitude = np.sqrt(Eye_Vector[0]*Eye_Vector[0]+Eye_Vector[1]*Eye_Vector[1]+Eye_Vector[2]*Eye_Vector[2])

	Gaze_angle = np.arccos(Gaze_Vector[0]/Gaze_magnitude)
	Eye_angle = np.arccos(Eye_Vector[0]/Eye_magnitude)

	Gaze_angle *= 180/np.pi
	Eye_angle *=180/np.pi

	if Gaze_angle>90:
		Gaze_angle-=90

	if Eye_angle>90:
		Eye_angle-=90




	return Gaze_angle-Eye_angle,Gaze_Vector


def draw_gaze(img, angle,pitch, tdx=None, tdy=None, size = 30):

	Gaze = -(angle * np.pi / 180)
	pitch = pitch*np.pi/180

 

    # Z-Axis (out of the screen) drawn in blue
    x = size * (sin(Gaze)) + tdx
    y = size * (-cos(Gaze) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x),int(y)),(255,0,0),2)



if __name__ == '__main__':
	main()