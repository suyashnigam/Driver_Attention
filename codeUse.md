# General guildlines for the code  

## src   

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




