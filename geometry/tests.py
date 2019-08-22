import numpy as np 
import time
from .EPnP import DLT as PoseDLT
from .EPnP import ComputeCost
from .EPnP import DisplayResults

def DLT_Projection():
	# load the data
	x=np.loadtxt('data/DLT_points2D.txt').T
	X=np.loadtxt('data/DLT_points3D.txt').T
	# compute the linear estimate with data normalization
	print ('Running DLT with data normalization')
	time_start=time.time()
	P_DLT = PoseDLT(x, X, normalize=True)
	cost = ComputeCost(P_DLT, x, X)
	time_total=time.time()-time_start
	# display the results
	print('took %f secs'%time_total)
	print('Cost=%.9f'%cost)
	displayResults(P_DLT, x, X, 'P_DLT')

K = np.array([[1545.0966799187809, 0, 639.5], 
      [0, 1545.0966799187809, 359.5], 
      [0, 0, 1]])

def P3P_Projection():
	pass