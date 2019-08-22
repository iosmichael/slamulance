import numpy as np
import cv2

from view import SLAMView
from feature import FeatureExtractor
from models import Frame, Feature, Pose
from utils import EssentialCamera

'''
Controller class that manages the data structure and view models
- settings
- algorithm banks
- camera intrinsics
'''
class SLAMController:

	def __init__(self, cap):
		self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) // 2)
		self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) // 2)
		self.total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
		self.camera = EssentialCamera(self.frame_width, self.frame_height, scale=1)

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
		
		frame, model = self.preprocessing_frame(frame)
		self.frames.append(model)

		if self.frame_idx - 1 < 0:
			self.view.draw_2d_frame(frame)
			self.frame_idx += 1
			return
		if self.frame_idx >= self.total_frame:
			# TODO: throw exceptions
			print("current frame out of bounds")
			return

		prev_model = self.frames[self.frame_idx - 1]
		# if we can find keypoints for both frames
		if prev_model.kps or model.kps:
			matches = self.feature_extractor.feature_matching(model.kps, model.des, prev_model.kps, prev_model.des)
			# shape of matches: 2 x n x 2
			# matches = self.normalize_matches(matches)
			self.view.draw_2d_matches(frame, matches)
		
			# clear keypoints and descriptors from the previous model after matching, memory efficiency
			prev_model.clear()
		self.view.draw_2d_frame(frame)
		self.frame_idx += 1

	# any preprocessing functionality here
	def preprocessing_frame(self, frame):
		frame_resize = cv2.resize(frame, dsize=(self.frame_width, self.frame_height))
		kps, des = self.feature_extractor.feature_detecting(frame_resize)
		model = Frame(kps, des)
		return frame_resize, model