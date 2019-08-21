import cv2
import numpy as np 

def get_K(frame_width, frame_height, scale = 1):
	'''
	camera intrinsic: F = [ f, 0, Cx
							0, f, Cy
							0, 0, 1  ]
	
	usually focal length is equivalent as the diagonal of the images
	'''
	Cx, Cy = frame_width // 2, frame_height // 2
	f = np.sqrt((frame_width/scale) ** 2 + (frame_height/scale) ** 2)
	K = np.array([[f, 0, Cx],
		[0, f, Cy],
		[0, 0, 1]])
	return K

def invK(K):
	return np.linalg.inv(K)

def normalize(pt):
	assert pt.shape == (2,1)
	norm_pt = np.ones((3,1))
	norm_pt[:2, 0] = pt
	return norm_pt

def denormalize(pt):
	pt /= pt[2, 0]
	return pt[:2, :]