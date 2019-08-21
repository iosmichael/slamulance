import numpy as np
from feature import get_K, invK, normalize, denormalize

if __name__ == '__main__':
	K = get_K(frame_width=1000, frame_height=500, f=525)
	invK = invK(K)
	pt = np.array([1,2]).reshape(2,1)
	print(normalize(pt))
	norm_pt = normalize(pt)
	print(denormalize(norm_pt))