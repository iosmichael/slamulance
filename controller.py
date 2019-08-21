import numpy as numpy
import cv2


'''
Controller class that manages the data structure and view models
- settings
- camera intrinsics
'''
class SLAMController:

	def __init__(self, file_path):
		self.file_path = file_path