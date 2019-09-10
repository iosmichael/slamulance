import numpy as np 
from geometry.utils import *

'''
each frame has many features, [many to many]
- it holds the inliers of matches
- it holds the keypoints and descriptors for only the next interations
'''
class Frame:

	def __init__(self, kps, des):
		self.keyframe = False
		# keypoints and descriptors
		self.kps = kps
		self.des = des
		self.leftInliers = None
		self.rightInliers = None
		# points are 3D points triangulated by points, saved as (kp_idx: Point3D)
		self.points = {}
		self.pose = None

	# return 3D points in the sequence with inlier indices
	def get_3D_points(self, inliers):
		pts = []
		for i in inliers:
			assert i in self.points.keys()
			pts.append(self.points[i])
		return pts

	def add_3D_point(self, inlier_idx, pt3D):
		self.points[inlier_idx] = pt3D

	def has_all(self, inlier_idx):
		for i in inlier_idx:
			if i not in self.points.keys():
				return False
		return True

	def clear(self):
		# release the memories once we are done with the triangulation with the next frame
		del self.des
		del self.leftInliers
		del self.rightInliers

'''
camera pose model
- holds the Rotation and Translation parameters
- holds the frame it is representing
'''
class Pose:

	def __init__(self, R, t):
		self.R = R
		self.t = t

	def P(self):
		return np.hstack((self.R, self.t))

	def Rt(self):
		return self.R, self.t

'''
3D feature model
- holds the pts and their respective frame id
- holds a triangulation value (3d homogeneous coordinates)
'''
class Point3D:

	def __init__(self, data):
		'''
		information on observations:
			points = [x ...]
					 [y ...]
		'''
		self.point2D = np.zeros((2, 0))
		self.frame_ids = []
		self.kps_idx = []

		# homogeneous 3D point
		assert data.shape == (4, 1)
		self.data = data

	def __str__(self):
		return "Point3D: [{}, {}, {}, {}]".format(
			self.data[0, 0].item(), 
			self.data[1, 0].item(), 
			self.data[2, 0].item(), 
			self.data[3, 0].item()
			)

	def add_observation(self, point, frame_idx):
		# points are represented by numpy data structure for computing efficiency
		assert point.shape == (2, 1)
		self.point2D = np.hstack((self.point2D, point))
		self.frame_ids.append(frame_idx)
		assert len(self.frame_ids) == self.point2D.shape[1]

	def get_observation(self, frame_idx):
		print(self.frame_ids)
		assert frame_idx in self.frame_ids
		return self.point2D[self.frame_ids.index(frame_idx)]

	def get_data(self):
		return self.data