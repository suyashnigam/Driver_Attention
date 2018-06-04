import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

import datasets, hopenet, utils
import eye_utils

from skimage import io
import dlib

########################## New Dependencies ###################################
import imutils
from imutils import face_utils 
########################## New Dependencies ###################################





def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--face_model', dest='face_model', help='Path of DLIB face detection model.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    parser.add_argument('--shape_model', dest='shape_model', help='Path of DLIB Face to landmark model.',
          default='', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True

    batch_size = 1
    gpu = args.gpu_id
    snapshot_path = args.snapshot
    out_dir = 'output/video'
    video_path = args.video_path
    shape_path = args.shape_model

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)

    ############################### Dlib face to landmark ############################
    face_to_landmark  = dlib.predictor(shape_path)
    ############################### Dlib face to landmark ############################


    print 'Loading snapshot.'
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print 'Loading data.'

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    video = cv2.VideoCapture(video_path)

    # New cv2
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, args.fps, (width, height))

    # # Old cv2
    # width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))   # float
    # height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)) # float
    #
    # # Define the codec and create VideoWriter object
    # fourcc = cv2.cv.CV_FOURCC(*'MJPG')
    # out = cv2.VideoWriter('output/video/output-%s.avi' % args.output_string, fourcc, 30.0, (width, height))

    txt_out = open('output/video/output-%s.txt' % args.output_string, 'w')

    frame_num = 1

    while frame_num <= args.n_frames:
        print frame_num

        ret,frame = video.read()
        if ret == False:
            break

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence






            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0); y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max); y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)

                #################### Img to grayscale #########################
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   ### Check this line
                color = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                #################### Img to grayscale #########################

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)

                assert img.size()[0] > 1 , 'Concatinate the image'

               

                #################### Face to eyes #############################
                shape = face_to_landmark(gray, det)
                landmarks = face_utils.shape_to_np(shape)
                #################### Face to eyes #############################



                #################### Detect Eyes  #############################
                right_eye = landmarks[36:42]
                left_eye = landmarks[42:48]
                #################### Detect Eyes ##############################

                #################### Bounding box eye #########################
                left,right = eye_utils.get_bounding_box(right_eye,left_eye,color)
                left = left.resize()
                right = right.resize()
                #################### Bounding box eye #########################

                #################### Conventional Eye Pose ####################
                # To do
                #################### Conventional Eye Pose ####################

                #################### Deep learning Eye Pose ###################
                # To do
                #################### Deep learning Eye Pose ###################




                yaw, pitch, roll = model(img)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.

                ##################### Check This #####################################
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
                ##################### Check This #####################################


                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

        
        
                cv2.putText(img = frame, text = "Roll:"+ str(roll_predicted), org = (100,50), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))         
                cv2.putText(img = frame, text = "Yaw:"+ str(yaw_predicted), org = (100,80), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))         
                cv2.putText(img = frame, text = "Pitch:"+ str(pitch_predicted), org = (100,110), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))         
        out.write(frame)
        frame_num += 1

    out.release()
    video.release()
