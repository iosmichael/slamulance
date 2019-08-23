'''
triangulation:
- link the old 3D points
- triangulate new 3D points
'''
def triangulation(f1, f2):
	# get old points
	kps = f1.kps
	# old ones: left inliers intersect right inliers
	intersect = np.logical_and(f1.leftInliers, f1.rightInliers)
	next_kps = f2.kps[f1.rightMatches[intersect]]
	curr_idx = np.arange(f1.kps.shape[0])[intersect]
	pts3D = []
	next_obs = []
	for i in range(curr_idx.shape[0]):
		pt3D = f1.points[i]
		next_obs.append(next_kps)
		pt3D.add_observation(next_kps, f2.idx)
		pts3D.append(pt3D)

	pose_estimation = get_pose(pts3D, next_obs)
	# pose estimation
	# new ones: left inliers diff right inliers
	diff = np.logical_xor(f1.rightInliers, intersect)
	triangulation(f1.kps[f1.rightInliers], f1.pose, f2.kps[f1.rightMatches], pose_estimation)
	