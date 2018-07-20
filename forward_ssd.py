"""
This is adapted from Tensorflow (https://github.com/tensorflow/models/tree/master/research/object_detection);
Save this code under the directory `models/research/object_detection/`
To use, run:
python tf_od_predict.py --model_name=building_od_ssd \
                         --path_to_label=data/building_od.pbtxt \
                         --test_image_path=test_images
"""

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

# sys.path.append("..")

# from utils import label_map_util
# from utils import visualization_utils as vis_util


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.shape
    return image.reshape((im_height, im_width, 1)).astype(np.uint8)

def tf_od_pred(detection_graph,img):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            print (image_tensor.shape)
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            # detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
               
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            width,height,c = img.shape

            img = np.repeat(img,3,axis=2)
            print (img.shape)
            image_np_expanded = np.expand_dims(img, axis=0)
            print (image_np_expanded.shape)

                # Actual detection.
            boxes,scores,num = sess.run(
                  [detection_boxes,detection_scores,num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                 # draw_bounding_box_on_image(image, boxes, )
                 # Visualization of the results of a detection.
                # vis_image = vis_util.visualize_boxes_and_labels_on_image_array(
                #          image_np,
                #          np.squeeze(boxes),
                #          np.squeeze(classes).astype(np.int32),
                #          np.squeeze(scores),
                #          category_index,
                #          use_normalized_coordinates=True,
                #          line_thickness=1)
            print("{} boxes in  image tile! of score {}".format(len(boxes),scores))
                # image_pil = Image.fromarray(np.uint8(vis_image)).convert('RGB')
                # with tf.gfile.Open(image_path, 'w') as fid:
                #      image_pil.save(fid, 'PNG')

            return np.squeeze(boxes),np.squeeze(scores)
            # boxes = np.squeeze(boxes)
            # ymin = int(boxes[0][0]*height)
            # xmin = int(boxes[0][1]*width)
            # ymax = int(boxes[0][2]*height)
            # xmax = int(boxes[0][3]*width)

            # cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(0,255,0),2)
            # cv2.imwrite('REKT.jpg',img)
            # print (boxes.shape)
            

def get_bbox_eye(img):
    # load your own trained model inference graph. This inference graph was generated from
    # export_inference_graph.py under model directory, see `models/research/object_detection/`
    # model_name = op.join(os.getcwd(), 'training') # Give definite path
    # Path to frozen detection graph.

    path_to_ckpt = os.path.join(os.getcwd(),'code/frozen_inference_graph.pb')# Give definite path
    # Path to the label file
    # path_to_label = op.join(os.getcwd(), FLAGS.path_to_label) 
    #only train on buildings
    num_classes = 1
    #Directory to test images path
    
    ############
    #Load the frozen tensorflow model
    #############

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    ############
    #Load the label file
    #############
    # label_map = label_map_util.load_labelmap(path_to_label)
    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)
    print (img.shape)
    img  = load_image_into_numpy_array(img)
    print (img.shape)

    return tf_od_pred(detection_graph,img)


def bbox_eye(img):

    # img = cv2.imread('image2.jpg',0)
    box,scores = get_bbox_eye(img)
    return box,scores
    