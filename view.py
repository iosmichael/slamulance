import cv2
import numpy as np 

class SLAMView:
	'''
	View class that contains all 2D cv2 drawing functionality in addition to 3D drawing
	'''
	def draw_2d_frame(self, frame, window_name='Driving POV'):
		cv2.imshow(window_name, frame)

	def draw_2d_matches(self, frame, matches):
		print("{} matches".format(matches.shape[1]))
		for i in range(matches.shape[1]):
			cv2.circle(frame, (matches[0, i, 0], matches[0, i, 1]), radius=3, color=(0, 255.0, 0))
			cv2.line(frame, (matches[0, i, 0], matches[0, i, 1]), (matches[1, i, 0], matches[1, i, 1]), (255, 0, 0), thickness=1)