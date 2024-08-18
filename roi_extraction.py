import glob
import os
import cv2
import numpy as np
import math
import time
import sys

from scipy.spatial import distance


def extract_roi_main(fn):

	# image file read 
	img = cv2.imread(fn)
		

	# image resize 
	resize_ratio = 0.25
	resize_color = cv2.resize(img, None, fx=resize_ratio, fy=resize_ratio, interpolation=cv2.INTER_AREA)
	rows, cols, chan = np.shape(resize_color)


	# find ROI region 
	height_length = 65 
	height_step = 30
		
	prev_mean = []
	max_idx = 0
	max_ed = 0
		
	for height_idx in range(0, 8):

		end_y = int(height_idx*height_step)+ int(height_length)-1

		if (height_idx*height_step + height_length) > rows:
			end_y = rows - 1
							
		roi_img = resize_color[int(height_idx*height_step):end_y, :]

		cur_mean = np.mean(roi_img, axis=(0,1))
			

		if height_idx > 0:
			ed = distance.euclidean(cur_mean, prev_mean)
			if ed > max_ed:
				max_ed = ed
				max_idx = height_idx

					
		prev_mean = cur_mean
		
	min_y = max_idx*height_step*4
	max_y = (max_idx*height_step + height_length)*4
		
	if min_y < 0: min_y = 0
			
	
	# ROI img extraction 
	roi_img = img[int(min_y):int(max_y), : ]
		
	cv2.imshow('roi image',roi_img)
	cv2.waitKey(0)

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print("roi_extraction.py img_file_name.jpg")
	else:
		extract_roi_main(sys.argv[1])
