import numpy as np
import cv2

img = cv2.imread("eye.jpg")
cv2.imshow("img",img)


cirlces  = img

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width, channels = img.shape 

print height,width
exit()

# circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,200)
ret,thresh1 = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)


# for i in circles[0,:]:
 	  
#    print i[0],i[1],i[2]  
#    cv2.circle(circles,(i[0],i[1]),i[2],(0,255,0),2)


cv2.imshow("circles",ret)
cv2.waitKey(0)

