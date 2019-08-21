import numpy as np
import cv2

from view import SLAMView
from feature import FeatureExtractor

'''
Controller class that manages the data structure and view models
- settings
- algorithm banks
- camera intrinsics
'''
class SLAMController:
	ASPECT_RATIO = 2 # reduce the original frame size by a scale of aspect ratio

	def __init__(self, cap):
		self.frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
		self.frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)

		self.view = SLAMView()
		self.feature_extractor = FeatureExtractor()

		self.frame_idx = 0
		self.frames = []


	def __str__(self):
		return "Controller: frames: width {} height {} total {}".format(self.frame_width, self.frame_height, self.total_frame)

	def process_frame(self, frame):
		'''
		main controller function that does basically everything
		'''
		# do nothing if it is the first frame
		if self.frame_idx - 1 < 0:
			self.frames.append(frame)
		else:

		self.frame_idx += 1

		# frame2 is the previous frame, which is the train frame
		# frame1 is the query frame
		frame1 = cv2.resize(frame1, dsize=(frame1.shape[1]//2, frame1.shape[0]//2))
		frame2 = cv2.resize(frame2, dsize=(frame2.shape[1]//2, frame2.shape[0]//2))
		kp1, des1 = feature_detecting(frame1)
		kp2, des2 = feature_detecting(frame2)
		matches = feature_matching(kp1, des1, kp2, des2)
		if kp1 is None or kp2 is None:
			return None, None
		draw_frame_annotation(frame1, matches=matches)

	def find_matches(self, frame):
		pass
