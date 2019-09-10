import cv2
import argparse

from controller import SLAMController
from view import SLAMView3D, SLAMView2D
import config

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='SLAM program with real world video')
	# additional input settings add here
	parser.add_argument('--video_path', default=config.VIDEO_PATH, type=str, help='video path, mp4 media type supported')
	args = parser.parse_args()
	
	cap = cv2.VideoCapture(args.video_path)
	if config.VIEW_2D:
		view2d = SLAMView2D()	
	if config.VIEW_3D:
		view3d = SLAMView3D()
	# initialize controller for processing
	SLAM = SLAMController(cap)

	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			'''
			this is our main function block for any SLAM operations
			'''
			image, matches = SLAM.process_frame(frame)
			if config.VIEW_2D:
				if image is not None:
					if matches is not None:
						view2d.draw_2d_matches(image, matches)
					view2d.draw_2d_frame(image)

			if len(SLAM.frames) <= 3:
				continue
			if config.VIEW_3D:
				view3d.draw_3d(SLAM.frames, SLAM.points)

			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break

	cap.release()
	cv2.destroyAllWindows()
