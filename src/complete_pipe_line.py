import sys, os, argparse

import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import GazeUtil
import datasets, hopenet, utils
import eye_utils

from skimage import io
import dlib

########################## New Dependencies ###################################
import forward_ssd
########################## New Dependencies ###################################


####################### Haar Cascade ##########################################
haar_cascade_path = os.path.join(os.getcwd(),'haarcascade_eye.xml')
eye_cascade = cv2.CascadeClassifier(haar_cascade_path)
####################### Haar Cascade ##########################################

image_path = os.path.join(os.getcwd(),'output/Images')
if not os.path.exists(image_path):
    os.makedirs(image_path)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head Pose + Gaze Direction')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--snapshot', dest='snapshot', help='Path of model snapshot.',
          default='', type=str)
    parser.add_argument('--video', dest='video_path', help='Path of video')
    parser.add_argument('--output_string', dest='output_string', help='String appended to output file')
    parser.add_argument('--n_frames', dest='n_frames', help='Number of frames', type=int)
    parser.add_argument('--fps', dest='fps', help='Frames per second of source video', type=float, default=30.)
    parser.add_argument('--eye_mode',dest='eye_mode',type=int,default=1,help='1 for Conventional 2 for deep learning')

    ####################### SSD based face detection ###########################################################
    parser.add_argument('--SSD_p', dest='SSD_protoxt',help='PAth to the SSD protoxt file')
    parser.add_argument('--SSD_m', dest='SSD_model',help='Path to the SSD model')
    parser.add_argument('--conf',dest='conf',type=float,default=0.5)
    ####################### SSD based face detection ###########################################################



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
    eye_mode = args.eye_mode

    ssd_p_path = args.SSD_protoxt
    ssd_m_path = args.SSD_model
    conf = args.conf

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(args.video_path):
        sys.exit('Video does not exist')

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

    # HOG based face detection model
    detector = dlib.get_frontal_face_detector()

    #SSD based face detectiom module
    ssd_net = cv2.dnn.readNetFromCaffe(ssd_p_path,ssd_m_path)

    print ('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print ('Loading data.')

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    model.cuda(gpu)

    print ('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(0,66)]
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

    frame_num = 0
    frames_detected = 0

    print ("Recording time taken")
    t = time.time()

    while frame_num <= args.n_frames:
        print (frame_num)

        ret,frame = video.read()
        if ret == False:
            break

        (w,h) = frame.shape[:2]

        ssd_pre_image = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,(300, 300), (104.0, 177.0, 123.0))

        cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        print("[INFO] computing object detections...")
        ssd_net.setInput(ssd_pre_image)
        dets = ssd_net.forward()

        for idx, det in enumerate(dets):
        

            confidence = dets[0, 0, idx, 2]
            print ("Confidence ",confidence)
            box = dets[0, 0, idx, 3:7] * np.array([h, w, h, w])
            (x_min, y_min, x_max, y_max) = box.astype("int")
            print (x_min,x_max,y_min,y_max)
           

            if confidence > conf:
                frames_detected +=1
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                img = cv2_frame[y_min:y_max,x_min:x_max]
                width_new, height_new = img.shape[:2]


                face_centre = ((x_min+x_max)/2,(y_min+y_max)/2)
                face_rad = (x_max - x_min)/2

                #################### Img to grayscale #########################
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   ### Check this line
                color = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                gray_equalized = cv2.equalizeHist(gray)
                #################### Img to grayscale #########################


                #################### Calculation of head pose #################
                img = Image.fromarray(img)
                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)


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

                #################### Calculation of head pose #################


                eye = []
                eye_rad = []
                xmin_eye = []
                ymin_eye = []
                eye_centre = []
                if eye_mode == 1:

                    num_eye = 0
                    objectsize_min = (60,60)
                    objectsize_max = (120,120)
                    eyes = eye_cascade.detectMultiScale(gray_equalized,scaleFactor=1.02,minNeighbors=3,minSize=objectsize_min,maxSize=objectsize_max)
                    for i,(ex,ey,ew,eh) in enumerate(eyes):
                        xmin_eye.append(ex + x_min)
                        ymin_eye.append(ey + y_min)
                        xmax_eye = xmin_eye[i] + ew
                        ymax_eye = ymin_eye[i] + eh
                        eye_rad.append(ew/2)
                        eye.append(frame[ymin_eye[i]:ymax_eye,xmin_eye[i]:xmax_eye])
                        eye_centre.append(((xmin_eye[i]+xmax_eye)/2,(ymin_eye[i]+ymax_eye)/2))
                        cv2.rectangle(frame,(xmin_eye[i],ymin_eye[i]),(xmax_eye,ymax_eye),(0,255,0),2)
                        # ymin_eye = ey+y_min
                        # xmin_eye = ex+x_min
                        print ('Eye detected')
                        num_eye+=1
            

                else:
                #################### Detect Eye deep learning  #################
                    print ("Using Deep Learning for eye detection")
                    boxes,scores = forward_ssd.bbox_eye(gray)
                    num_eye = 0
                    height_gray = y_max - y_min
                    width_gray  = x_max - x_min
                    for i in range(0,2):
                          if scores[i]>0.1:
                              print ('Eye Detected')
                              ymin_eye.append(int(boxes[i][0]*height_gray) + y_min)
                              xmin_eye.append(int(boxes[i][1]*width_gray)  + x_min)
                              ymax_eye = int(boxes[i][2]*height_gray) + y_min
                              xmax_eye = int(boxes[i][3]*width_gray)  + x_min         
                              cv2.rectangle(frame,(xmin_eye[i],ymin_eye[i]),(xmax_eye,ymax_eye),(0,255,0),2)
                              eye.append(frame[ymin_eye[i]:ymax_eye,xmin_eye[i]:xmax_eye])
                              eye_centre.append(((xmin_eye[i]+xmax_eye)/2,(ymin_eye[i]+ymax_eye)/2))
                              num_eye+=1
                              eye_rad.append((xmax_eye-xmin_eye[i])/2)
                eye = np.array(eye)
               

                #################### Gaze Calculation ###########################
                for i in range(0,num_eye):
                    gaze_x,gaze_y = GazeUtil.getGaze(eye[i])
                    gaze_y+= ymin_eye[i]
                    gaze_x+= xmin_eye[i]
                    cv2.circle(frame, (gaze_x,gaze_y), 5, 255, -1)
                    gaze_centre = (gaze_x,gaze_y)
                    gaze_angle,gaze_vector = GazeUtil.GazeAngle(eye_centre[i],gaze_centre,eye_rad[i])
                    GazeUtil.draw_gaze(frame, gaze_angle[0],gaze_angle[1], tdx = gaze_centre[0], tdy= gaze_centre[1],marker=0)
                    cv2.putText(img = frame, text = "Gaze: "+ str(gaze_x)+','+str(gaze_y), org = (100,30), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))         
                    cv2.putText(img = frame, text = "Gaze Angle: "+ str(gaze_angle),org = (100,150+30*i), fontFace = cv2.FONT_HERSHEY_DUPLEX,fontScale = 1, color = (0, 0, 0)) 
                ######################## Geometric Calculations ####################



             

                # Print new frame with cube and axis
                txt_out.write(str(frame_num) + ' %f %f %f\n' % (yaw_predicted, pitch_predicted, roll_predicted))
                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)

                # Plot expanded bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                cv2.putText(img = frame, text = "Roll:"+ str(roll_predicted), org = (100,60), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))         
                cv2.putText(img = frame, text = "Yaw:"+ str(yaw_predicted), org = (100,90), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0))         
                cv2.putText(img = frame, text = "Pitch:"+ str(pitch_predicted), org = (100,120), fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 1, color = (0, 0, 0)) 
            else:
                print ("Not Detected")        
        out.write(frame)
        frame_num += 1
    txt_out.write("Meta Data: The confidence level was {} and the number of frames detected was {} out of {}, the time taked {}".format(conf,frames_detected,frame_num-1,time.time()-t))
    print ('{} out of {} frames detected in time {}'.format(frames_detected-1,frame_num-1,time.time()-t))
    out.release()
    video.release()
