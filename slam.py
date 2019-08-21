import cv2
import numpy as np
from feature import get_K

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

def feature_detecting(frame):
	orb = cv2.ORB_create()
	pts = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=7)
	if pts is None:
		return None, None
	kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in pts]
	kps, des = orb.compute(frame, kps)
	return kps, des

def feature_matching(kp1, des1, kp2, des2):
	# BFMatcher with default params
	# Referred from OpenCV Website
	if des1 is None or des2 is None:
		return []
	bf = cv2.BFMatcher(cv2.NORM_HAMMING)
	matches = bf.knnMatch(des1, des2, k=2)
	# Apply ratio test
	good = []
	cnt = 100
	for m,n in matches:
		if m.distance < 0.75*n.distance:
			if m.distance < 32:
				good.append((kp1[m.queryIdx], kp2[m.trainIdx]))

	# apply ransac
	return good

def draw_frame_annotation(frame, matches):
	# transform matches
	matches_np = np.array([[np.array(kp1.pt), np.array(kp2.pt)] for kp1, kp2 in matches])
	print("{} matches".format(matches_np.shape[0]))
	
	# testing the intrinsic K matrix
	# K = get_K(1920//2, 1080//2, 525)
	# pts1 = matches_np[:, 0, :].transpose((0, 2, 1))
	# pts2 = matches_np[:, 1, :].transpose((0, 2, 1))
	# normalize and denormalize with K

	# for i in range(pts.shape[0]):
	# 	cv2.circle(frame, (pts[i, :, 0], pts[i, :, 1]), radius=3, color=(0, 255.0, 0))
	for kp1, kp2 in matches:
		cv2.circle(frame, (int(kp1.pt[0]), int(kp1.pt[1])), radius=3, color=(0, 255.0, 0))
		cv2.line(frame, (int(kp1.pt[0]), int(kp1.pt[1])), (int(kp2.pt[0]), int(kp2.pt[1])), (255, 0, 0), thickness=1)
	cv2.imshow('Driving POV', frame)

def processing_frames(frame1, frame2):
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

if __name__ == '__main__':
	cap = cv2.VideoCapture('driving.mp4')
	print('width: {} height: {} frames: {}'.format(cap.get(cv2.CAP_PROP_FRAME_WIDTH),
		cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
		cap.get(cv2.CAP_PROP_FRAME_COUNT)))
	pre_frame = None
	while cap.isOpened():
		ret, frame = cap.read()
		if ret == True:
			if pre_frame is None:
				pre_frame = frame
			else:
				# match_frames(frame, pre_frame)
				processing_frames(frame, pre_frame)
				pre_frame = frame
			if cv2.waitKey(25) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	cv2.destroyAllWindows()
