import os
import io
import glob
import hashlib
import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf
import random

from PIL import Image
import dataset_util

def create_example(xml_file):
        #process the xml file
        tree = ET.parse(xml_file)
        root = tree.getroot()

        image_name = root.find('filename').text
        file_name = image_name.encode('utf8')

        size = root.find('size')
        width = int(size[0].text)
        height = int(size[0].text)

        xmin=[]
        ymin=[]
        xmax=[]
        ymax=[]
        xcentre=[]
        ycentre=[]

        classes =[]
        classes_text =[]

        for member in root.findall('object'):
            classes_text.append('Eye'.encode('utf8'))

            xmin.append(float(member[1][0].text)/width)
            xmax.append(float(member[1][2].text)/width)
            ymin.append(float(member[1][1].text)/height)
            ymax.append(float(member[1][3].text)/height)

            xcentre.append(float(member[2][0].text)/width)
            ycentre.append(float(member[2][1].text)/height)


            classes.append(1)

        full_path = os.path.join('./JPEGImages', '{}'.format(image_name))  #provide the path of images directory
        with tf.gfile.GFile(full_path, 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        key = hashlib.sha256(encoded_jpg).hexdigest()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(file_name),
            'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
            'image/object/centre/x':dataset_util.float_list_feature(xcentre),
            'image/object/centre/y':dataset_util.float_list_feature(ycentre),

        })) 
        return example  



def main():
    writer_train = tf.python_io.TFRecordWriter('train.record')     
    writer_test = tf.python_io.TFRecordWriter('test.record')
    #provide the path to annotation xml files directory
    filename_list=tf.train.match_filenames_once("./Annotations/*.xml")
    init = (tf.global_variables_initializer(), tf.local_variables_initializer())
    sess=tf.Session()
    sess.run(init)
    list=sess.run(filename_list)
    random.shuffle(list)   #shuffle files list
    i=1 
    tst=0   #to count number of images for evaluation 
    trn=0   #to count number of images for training
    for xml_file in list:
      example = create_example(xml_file)
      if (i%10)==0:  #each 10th file (xml and image) write it for evaluation
         writer_test.write(example.SerializeToString())
         tst=tst+1
      else:          #the rest for training
         writer_train.write(example.SerializeToString())
         trn=trn+1
      i=i+1
      print(xml_file)
    writer_test.close()
    writer_train.close()
    print('Successfully converted dataset to TFRecord.')




if __name__== '__main__':
    main()