import numpy as np
import cv2
import random
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
from numpy import genfromtxt
import os

def plot(l,r):
	save_dir = './JPEGImages/'
	xml_dir = './Annotations/'

	save_dir = os.path.join(os.getcwd(),save_dir)
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	xml_dir = os.path.join(os.getcwd(),xml_dir)
	if not os.path.exists(xml_dir):
		os.makedirs(xml_dir)

	CSV_COL = ['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','left_eye_inner_corner_x','left_eye_inner_corner_y','left_eye_outer_corner_x','left_eye_outer_corner_y','right_eye_inner_corner_x','right_eye_inner_corner_y','right_eye_outer_corner_x','right_eye_outer_corner_y','left_eyebrow_inner_end_x','left_eyebrow_inner_end_y','left_eyebrow_outer_end_x','left_eyebrow_outer_end_y','right_eyebrow_inner_end_x','right_eyebrow_inner_end_y','right_eyebrow_outer_end_x','right_eyebrow_outer_end_x','Image']
	data = pd.read_csv("training.csv")
	data = data.dropna(axis=0)
	j = 0
	while j<len(data):
		image_all = np.array(data[CSV_COL[-1]].values)
		left_x_min_all = np.array(data['left_eye_inner_corner_x'].values) - l*(np.abs(np.array(data['left_eye_inner_corner_x'].values)-np.array(data['left_eye_outer_corner_x'])))
		left_y_min_all = np.array(data['left_eye_inner_corner_y'].values) - r*(np.abs(np.array(data['left_eyebrow_inner_end_y'].values)-np.array(data['left_eye_inner_corner_y'])))
		left_x_max_all = np.array(data['left_eye_outer_corner_x'].values) + l*(np.abs(np.array(data['left_eye_inner_corner_x'].values)-np.array(data['left_eye_outer_corner_x'])))
		left_y_max_all = np.array(data['left_eye_outer_corner_y'].values) + r*(np.abs(np.array(data['left_eyebrow_inner_end_y'].values)-np.array(data['left_eye_inner_corner_y']))) 
		# print image_all.shape
		right_x_min_all = np.array(data['right_eye_outer_corner_x'].values) - l*(np.abs(np.array(data['right_eye_inner_corner_x'].values)-np.array(data['right_eye_outer_corner_x'])))
		right_y_min_all = np.array(data['right_eye_outer_corner_y'].values) - r*(np.abs(np.array(data['right_eyebrow_inner_end_y'].values)-np.array(data['right_eye_inner_corner_y'])))
		right_x_max_all = np.array(data['right_eye_inner_corner_x'].values) + l*(np.abs(np.array(data['right_eye_inner_corner_x'].values)-np.array(data['right_eye_outer_corner_x'])))
		right_y_max_all = np.array(data['right_eye_inner_corner_y'].values) + r*(np.abs(np.array(data['right_eyebrow_inner_end_y'].values)-np.array(data['right_eye_inner_corner_y']))) 

		left_eye_center_x_all = np.array(data['left_eye_center_x'].values)
		right_eye_center_x_all = np.array(data['right_eye_center_x'].values)
		left_eye_center_y_all = np.array(data['left_eye_center_y'].values)
		right_eye_center_y_all = np.array(data['right_eye_center_y'].values)

		# i = random.randint(0,len(image_all))
		i = j

		image = np.fromstring(image_all[i],sep=' ')
		# image = image/255
		image = np.reshape(image,(96,96))


		left_x_min = int(left_x_min_all[i])
		left_x_max = int(left_x_max_all[i])
		left_y_max = int(left_y_max_all[i])
		left_y_min = int(left_y_min_all[i])


		right_x_min = int(right_x_min_all[i])
		right_x_max = int(right_x_max_all[i])
		right_y_max = int(right_y_max_all[i])
		right_y_min = int(right_y_min_all[i])

		left_center_x = int(left_eye_center_x_all[i])
		left_center_y = int(left_eye_center_y_all[i])
		right_center_x = int(right_eye_center_x_all[i])
		right_center_y = int(right_eye_center_y_all[i])

		cv2.imwrite(save_dir+str(j)+".jpg",image)

		# cv2.waitKey()

		Annotations = ET.Element('Annotations')
		ET.SubElement(Annotations, "filename").text = str(j) + '.jpg'
		ET.SubElement(Annotations,"source").text = save_dir + str(j) + '.jpg'
		size = ET.SubElement(Annotations,'size')
		ET.SubElement(size,"width").text = '96'
		ET.SubElement(size,"Height").text = '96'
		ET.SubElement(size,"Channel").text = '1'
		obj = ET.SubElement(Annotations,'object',name="Left_eye")
		name = ET.SubElement(obj,'name')
		name.text = 'Eye'
		bbox = ET.SubElement(obj,'bbox')
		ET.SubElement(bbox,"xmin").text = str(left_x_min)
		ET.SubElement(bbox,'ymin').text = str(left_y_min)
		ET.SubElement(bbox,'xmax').text = str(left_x_max)
		ET.SubElement(bbox,'ymax').text = str(left_y_max)

		eye_centre = ET.SubElement(obj,'eye_centre')
		ET.SubElement(eye_centre,'x').text = str(left_center_x)
		ET.SubElement(eye_centre,'y').text = str(left_center_y)



		obj = ET.SubElement(Annotations,'object',name="Right_eye")
		name = ET.SubElement(obj,'name')
		name.text = 'Eye'
		bbox = ET.SubElement(obj,'bbox')
		ET.SubElement(bbox,"xmin").text = str(right_x_min)
		ET.SubElement(bbox,'ymin').text = str(right_y_min)
		ET.SubElement(bbox,'xmax').text = str(right_x_max)
		ET.SubElement(bbox,'ymax').text = str(right_y_max)

		eye_centre = ET.SubElement(obj,'eye_centre')
		ET.SubElement(eye_centre,'x').text = str(right_center_x)
		ET.SubElement(eye_centre,'y').text = str(right_center_y)

		tree = ET.ElementTree(Annotations)
		tree.write(xml_dir +str(j)+'.xml')


		j+=1



def parse_args():
	 parser = argparse.ArgumentParser(description="Ratios")
	 parser.add_argument('--l',dest='l',type=float,default=0.2)
	 parser.add_argument('--r',dest='r',type=float,default=0.8)

	 args = parser.parse_args()
	 return args

if __name__ == '__main__':
    args = parse_args()
    l = args.l
    r = args.r
    plot(l,r)