import os
import numpy as np
import cv2

FILE_PATH = 'test/templeSparseRing'
CAMERA_DATA = 'templeSR_par.txt'

def get_camera_data(content):
	length = int(content[0])
	del content[0]
	params = {l.split(' ')[0]: np.array(l.split(' ')[1:]).astype(float) for l in content}
	assert length == len(params.keys())
	for key in params.keys():
		K = params[key][:9].reshape(3,3)
		R = params[key][9:18].reshape(3,3)
		t = params[key][18:].reshape(3,1)
		params[key] = {"K": K, "R": R, "t": t}
	return params

def get_image_data(params):
	images = {key: cv2.imread(os.path.join(FILE_PATH, key)) for key in params.keys()}
	return images

def get_stereo_data():
	data, images, image_keys = None, None, None
	with open(os.path.join(FILE_PATH, CAMERA_DATA), 'r') as f:
		content = f.readlines()
		data = get_camera_data(content)
		images = get_image_data(data)
		image_keys = [name for name in data.keys()]
	return data, images, image_keys

def main():
	with open(os.path.join(FILE_PATH, CAMERA_DATA), 'r') as f:
		content = f.readlines()
		data = get_camera_data(content)
		print('data length: {}'.format(len(data.keys())))
		images = get_image_data(data)
		sample_key = None
		for k in data.keys():
			sample_key = k
			break
		print('K: {}, R: {}, t: {}'.format(data[sample_key]["K"], data[sample_key]["R"], data[sample_key]["t"]))
		print('shapes: K: {}, R: {}, t: {}'.format(data[sample_key]["K"].shape, data[sample_key]["R"].shape, data[sample_key]["t"].shape))
		print(np.linalg.det(data[sample_key]["R"]))

if __name__ == '__main__':
	main()