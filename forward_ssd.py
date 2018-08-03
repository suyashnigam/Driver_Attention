
import os
import sys
import glob
import cv2
import six.moves.urllib as urllib
import tensorflow as tf
import tarfile

from io import StringIO
import zipfile
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from PIL import ImageDraw, Image

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.shape
    return image.reshape((im_height, im_width, 1)).astype(np.uint8)

def tf_od_pred(detection_graph,img):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            width,height,c = img.shape

            img = np.repeat(img,3,axis=2)
            print (img.shape)
            image_np_expanded = np.expand_dims(img, axis=0)
            print (image_np_expanded.shape)

                # Actual detection.
            boxes,scores,num = sess.run(
                  [detection_boxes,detection_scores,num_detections],
                  feed_dict={image_tensor: image_np_expanded})
            

            return np.squeeze(boxes),np.squeeze(scores)
           
            

def get_bbox_eye(img):
   

    path_to_ckpt = os.path.join(os.getcwd(),'frozen_inference_graph.pb')# Give definite path
    
    num_classes = 1
   
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')


    img  = load_image_into_numpy_array(img)
    return tf_od_pred(detection_graph,img)


def bbox_eye(img):

   
    box,scores = get_bbox_eye(img)
    return box,scores
    