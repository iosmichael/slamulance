import cv2

class SLAMView:
	'''
	View class that contains all 2D cv2 drawing functionality in addition to 3D drawing
	'''
	def __init__(self):
		
		pass

	def draw_2d_frame(frame, window_name='Driving POV'):
		cv2.imshow(window_name, frame)

	def draw_2d_matches(frame, matches):
		matches_np = np.array([[np.array(kp1.pt), np.array(kp2.pt)] for kp1, kp2 in matches])
		print("{} matches".format(matches_np.shape[0]))
		for kp1, kp2 in matches:
			cv2.circle(frame, (int(kp1.pt[0]), int(kp1.pt[1])), radius=3, color=(0, 255.0, 0))
			cv2.line(frame, (int(kp1.pt[0]), int(kp1.pt[1])), (int(kp2.pt[0]), int(kp2.pt[1])), (255, 0, 0), thickness=1)