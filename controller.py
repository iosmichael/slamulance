import numpy as np
import cv2

from view import SLAMView
from feature import FeatureExtractor
from models import Frame, Point3D, Pose
from geometry.EPnP import DLT as Pose_DLT
from geometry.fundamental import DLT_F, DLT_E, decompose_E
from geometry.utils import Homogenize, Dehomogenize

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
		print('WIDTH: {}, HEIGHT: {}, FRAME_COUNT: {}'.format(self.frame_width, self.frame_height, self.total_frame))

		self.view = SLAMView()
		self.feature_extractor = FeatureExtractor()

		self.frame_idx = 0
		self.frames = []
		self.K = np.array([[525, 0, self.frame_width//2],
						   [0, 525, self.frame_height//2],
						   [0, 0, 1]])

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
		if prev_model.kps is not None and model.kps is not None:
			
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
				
				# self.PoseEstimation(model, prev_model)
				
				# triangulation
				
				# self.Triangulation(model, prev_model)
				pass

			# shape of matches: 2 x n x 2
			# post-processing the keypoints data
			kp1 = model.kps[model_inliers]
			kp2 = prev_model.kps[prev_inliers]
			matches = np.stack((kp1, kp2), axis=0)
			self.view.draw_2d_matches(frame, matches)
		
			# clear keypoints and descriptors from the previous model after matching, memory efficiency
			# prev_model.clear()
		self.view.draw_2d_frame(frame)
		self.frame_idx += 1

	# any preprocessing functionality here
	def preprocessing_frame(self, frame):
		frame_resize = cv2.resize(frame, dsize=(self.frame_width, self.frame_height))
		kps, des = self.feature_extractor.feature_detecting(frame_resize)
		kps = np.array([item.pt for item in kps])
		print('changing keypoints to np array')
		model = Frame(kps, des)
		return frame_resize, model

	# DLT estimation for Projective Matrix P, 
	# given 3D points from previous frames and 2D points in current frames
	def PoseEstimation(self, f1, f2):
		pts3D = f2.get_3D_points(f2.rightInliers)
		pts2D = f1.kps[f1.leftInliers]
		X = np.array([item.data() for item in pts3D])
		print('X shape: {}'.format(X.shape))
		x = np.array([item.pt for item in pts2D])
		print('x shape: {}'.format(x.shape))
		print('Pose estimation begins here')

	# Triangulation for new 3D points
	# given newly matched points in both frames
	def Triangulation(self, f1, f2):
		# creation of new 3D points
		# intersection of two inliers
		print('Triangulation begins here')
		pass

	def TwoViewTriangulation(self, f1, f2):
		# creation of fundamental matrix and 3D points assuming the first pose (f2) is [I | 0]
		# normalized_pt1 = np.linalg.inv(K) @ f1.kps[f1.leftInliers]
		# normalized_pt2 = np.linalg.inv(K) @ f2.kps[f2.rightInliers]
		print('matches: {}'.format(len(f2.rightInliers)))
		# save for testing
		np.save('./test/pts1.npy', f2.kps[f2.rightInliers])
		np.save('./test/pts2.npy', f1.kps[f1.leftInliers])

		E = DLT_E(x1=f2.kps[f2.rightInliers], x2=f1.kps[f1.leftInliers], K=self.K)
		P, P2 = decompose_E(E, f2.kps[f2.rightInliers][0], f1.kps[f1.leftInliers][0])
		print('First camera R: {} t: {}'.format(P[:, :3], P[:, -1]))
		print('Second camera R: {} t: {}'.format(P2[:, :3], P2[:, -1]))
		f2.pose = Pose(P[:, :3], P[:, -1])
		f1.pose = Pose(P2[:, :3], P2[:, -1])
		print('TwoViewTriangulation begins here')
		pass

