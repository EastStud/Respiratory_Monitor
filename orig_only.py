#import the necessary modules
import freenect
import cv2
import numpy as np
import time as t
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage.filters as filt
import scipy.signal as sig
import pickle
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array
 
if __name__ == "__main__":
    samples = 1800  # Determine length based on 30 frames / samples per second
                    # This is for 60 seconds at 30 samples per second
    myArrayOfImages = [] #original
    timeStart = t.time()
    while len(myArrayOfImages) < samples:
	#Append original depth array to array of depths
        depth = get_depth()
        myArrayOfImages.append(depth)

        #arraySum = sum(depth)
	print(len(myArrayOfImages))
        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
   	#cv2.destroyAllWindows()
    timeEnd = t.time()
    
    #Pickle data for later use
    output = open('some_samples.pkl','wb')
    pickle.dump(myArrayOfImages, output)
    output.close()
    











