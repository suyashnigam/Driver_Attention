import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

from operator import itemgetter


def get_bounding_box(right,left,image):

	x_max_left = max(left,key=itemgetter(1))[0]
	x_min_left = min(left,key=itemgetter(1))[0]
	y_max_left = max(left,key=itemgetter(1))[1]
	y_min_left = min(left,key=itemgetter(1))[1]

	x_max_right = max(right,key=itemgetter(1))[0]
	x_min_right = min(right,key=itemgetter(1))[0]
	y_max_right = max(right,key=itemgetter(1))[1]
	y_min_right = min(right,key=itemgetter(1))[1]

	



