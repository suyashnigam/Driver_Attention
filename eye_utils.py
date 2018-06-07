import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from operator import itemgetter


def get_bounding_box(right,left,image):


	left_y = np.zeros(6)
	left_x = np.zeros(6)
	right_x = np.zeros(6)
	right_y = np.zeros(6)
	for i in range(6):
		right_x[i] = int(right[i,0])
		left_x[i]  = int(left[i,0])
		right_y[i] = int(right[i,1])
		left_y[i]  = int(left[i,1])
  
	return [np.amin(left_x),np.amin(left_y),np.amax(left_x),np.amax(left_y)],[np.amin(right_x),np.amin(right_y),np.amax(right_x),np.amax(right_y)]



def deviation(img):


	ret,thresh = cv2.threshold(img,127,255,0)
	im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


	
def shape_to_landmarks(shape):
	landmarks = np.zeros((shape.num_parts,2))

	for i in range(0,shape.num_parts):
		landmarks[i,0] = int(shape.part(i).x)
		landmarks[i,1] = int(shape.part(i).y)

	return landmarks
