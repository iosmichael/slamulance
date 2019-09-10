import numpy as np
from geometry.utils import *
from geometry.EPnP import *

def main():
	# load data
	x0 = np.loadtxt('test/hw2_points2D.txt').T
	X0 = np.loadtxt('test/hw2_points3D.txt').T
	K = np.array([[1545.0966799187809, 0, 639.5], 
				  [0, 1545.0966799187809, 359.5], 
				  [0, 0, 1]])
	x0_hat = np.linalg.inv(K) @ Homogenize(x0)
	print('x shape', x0_hat.shape)
	print('X shape', X0.shape)
	P = EPnP(x0_hat, X0)
	print("det(R): {}".format(np.linalg.det(P[:, :3])))
	print("R: {}".format(P[:, :3]))
	print("t: {}".format(P[:, -1]))
	print("pose estimation reprojection error: {}".format(ComputeCost(P, x0_hat, X0)))
	print('Passed Absolute Pose Estimation with EPnP')

if __name__ == '__main__':
	main()