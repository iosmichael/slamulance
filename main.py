import cv2
import argparse

from controller import SLAMController
from view import SLAMView3D

VIDEO_PATH = './videos/driving_country.mp4'

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SLAM program with real world video')
	# additional input settings add here
	parser.add_argument('--video_path', default=VIDEO_PATH, type=str, help='video path, mp4 media type supported')
	args = parser.parse_args()
	
	cap = cv2.VideoCapture(args.video_path)
	view3d = SLAMView3D()
	# initialize controller for processing
	SLAM = SLAMController(cap)
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			'''
			this is our main function block for any SLAM operations
			'''
			SLAM.process_frame(frame)
			if len(SLAM.frames) <= 3:
				continue
			view3d.draw_3d(SLAM.frames, SLAM.points)
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
