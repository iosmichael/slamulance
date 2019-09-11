'''
Visual Monocular SLAM Implementation
Created: Sept 8, 2019
Author: Michael Liu (GURU AI Group, UCSD)
'''

import cv2
import numpy as np
import time
import os

'''
CONFIGURATION FOR CLIPPING VIDEO
'''
FILE_PATH = './driving.mp4'
SAVE_PATH = './driving_clip.mp4'
START_FRAME = 1000
END_FRAME = 3000

if __name__ == '__main__':

    if os.path.exists(FILE_PATH):
        cap = cv2.VideoCapture(FILE_PATH)
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
            print("Error opening video stream or file")

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Frame Length: {}'.format(length))

        # Pre-allocate number of split
        f_index = 0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        FPS = int(cap.get(cv2.CAP_PROP_FPS))
        print('ORIGINAL FPS: {}'.format(FPS))
        print('IMG SHAPE ({},{})'.format(FRAME_WIDTH, FRAME_HEIGHT))
        writer = cv2.VideoWriter(SAVE_PATH, fourcc, FPS, (FRAME_WIDTH, FRAME_HEIGHT))
        data = []
        while(cap.isOpened()):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret == True:
                if f_index < END_FRAME and f_index >= START_FRAME:
                    writer.write(frame)
                if f_index > END_FRAME:
                    break
                f_index += 1
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                print('End of Frame, Saving Video')
                break
        # When everything done, release the video capture object
        print('Finished Clipping Video')
        cap.release()
        writer.release()
        # Closes all the frames
        cv2.destroyAllWindows()
    else:
        print('{}: file cannot find'.format(FILE_PATH))