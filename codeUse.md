# General guildlines for the code  

## Master script   

#### complete_pipe_line.py   

* Parser  
> python complete_pipe_line.py --snapshot path_to_hopenetPkl --video path_to_video --output_string xyz --n_frames 60 --fps 30 --SSD_p ~/Documents/Suyashn/intern/deploy.prototxt.txt --SSD_m ~/Documents/Suyashn/intern/res10_300x300_ssd_iter_140000.caffemodel --conf 0.2 --eye_mode 1  

1. snapshot: Path to the pickle file of saved hopenet weights.   
2. video   : Path to the video to be processed.  
3. output_string : The string prefixed to outputed processed video.  
4. n_frames      : Number of frames to be processed.  
5. fps           : Fps of video.  
6. SSD_p         : Protoxt.txt file of trained caffemodel for face detection.  
7. SSD_m         : Saved model of caffemodel for face detection.  
8. conf          : Threshold confidence for detection, 0.5 for normal, 0.2 for glasses.  
9. eye_mode      : 1 for conventional 2 for face detection(needs a more trained model)  

* Dependencies  

1. PyTorch  
2. Tensorflow > 1.4  
3. OpenCV 3.0  

## Eye Deep Learning  
There is a frozen_inference_grph.pb from a partially trained model, use tensorflow's object detection module to generate .pb model to replace.

#### forward_ssd.py
Takes in cropped face image from face detection model and detects eye using the trained deep learning model. (loads a .pb file for inference)

#### ssd_config.config
The parameters for training the module

## Gaze Detection
#### python_iris.py  
This script calculates the iris coordinates using a cropped out eye image.  
Hyperparameters:-  (default)
1. Blur_size : Kernel size of Gaussian Blur (3)
2. PostConstant = The threshold fraction above which pixels are not considered. (0.97)
3. thresholdGradient = Threshold value of gradient, below which are not considered in voting. (50)
4. prepro  = Boolean for pre processing or not (True).
5. postpro = Boolean for pose processsing or not (True).
6. new_width = Width of scaled down image for detection, a smaller image leads to faster computation. (20)

## Utilities  

#### dataset_util.py  

To create an object detection database from kaggle facial landmark database.  
* Parser
> python dataset_util.py --l 0.8 --r 0.2  
  
l and r describe the eye bbox heurestics.