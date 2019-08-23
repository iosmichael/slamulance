import numpy as np
import cv2

from view import SLAMView
from feature import FeatureExtractor
from models import Frame, Feature, Pose

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
			
			# indices for matched keypoints
			model_inliers, prev_inliers = self.feature_extractor.feature_matching(model.kps, model.des, prev_model.kps, prev_model.des)

			# update connection graph between the two frames
			prev_model.rightInliers = prev_inliers
			model.leftInliers = model_inliers
			
			if prev_model.pose is None:
				# use matches to calculate fundamental matrix
				# perform triangulation with P = [I | 0] and P' = [M | v]
				self.TwoViewTriangulation(model, prev_model)
			else:
				# find the 3D points in the previous frame
				# DLT for pose estimation
				self.PoseEstimation(model, prev_model)
			# triangulation
			self.Triangulation(model, prev_model)

			# shape of matches: 2 x n x 2
			# post-processing the keypoints data
			kp1 = np.array([item.pt for item in model.kps[model_inliers]])
			kp2 = np.array([item.pt for item in prev_model.kps[prev_inliers]])
			matches = np.stack((kp1, kp2), axis=0)
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

	# DLT estimation for Projective Matrix P, 
	# given 3D points from previous frames and 2D points in current frames
	def PoseEstimation(self, f1, f2):
		pts3D = f2.get_3D_points(f2.rightInliers)
		pts2D = f1.kps[f1.leftInliers]
		print('Pose estimation begins here')

	# Triangulation for new 3D points
	# given newly matched points in both frames
	def Triangulation(self, f1, f2):
		# creation of new 3D points
		pass

	def TwoViewTriangulation(self, f1, f2):
		# creation of fundamental matrix and 3D points assuming the first pose is [I | 0]
		pass

