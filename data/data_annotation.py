from scipy import misc as sci
import numpy as np
import os
import math
import cv2

# data_dir = 'data_collection/'
data_dir = 'round1/'
RADIUS = 10
# fourcc = cv2.VideoWriter_fourcc(*"XVID")
# fourcc = cv2.VideoWriter_fourcc(*"MJPG")


# get sub-directories list
for root, dirs, files in os.walk(data_dir):
	sub_dirs = dirs
	break
# print(sub_dirs)

# read files under each sub-dir
for sub_dir in sub_dirs:
	print(sub_dir)
	for root, dirs, files in os.walk(data_dir + sub_dir):
		break
	print(len(files))

	# mkdir draw folder
	img_dir_path = data_dir + sub_dir + '/' + 'draw/'
	img_dir = os.path.dirname(img_dir_path)
	# if os.path.exists(img_dir):
	# 	continue
	if not os.path.exists(img_dir):
		os.makedirs(img_dir)

	num_img = len(files) - 6
	gaze_file_path = data_dir + sub_dir + '/' + sub_dir + 'testfile.txt'
	if not os.path.exists(gaze_file_path):
		continue
	gaze_file = open(gaze_file_path)


	# video_file_path = data_dir + sub_dir + '/' + 'video.avi'
	height, width = 360, 640
	# video_file = cv2.VideoWriter(video_file_path, fourcc, 30, (width, height))

	for i in range(0, num_img):
		img_path = data_dir + sub_dir + '/' + sub_dir + str("%06d" % i) + '.jpg'
		img_dst_path = data_dir + sub_dir + '/draw/' + str("%06d" % i) + '.jpg'
		poss = float(gaze_file.readline())

		if not os.path.exists(img_path):
			gaze_file.readline()
			gaze_file.readline()
			continue
		img = cv2.imread(img_path)

		x = math.floor(float(gaze_file.readline()) * img.shape[1])
		y = math.floor((1-float(gaze_file.readline())) * img.shape[0])



		# print(img_path)
		if x <= img.shape[1] and x >= 0 and y <= img.shape[0] and y >= 0:
			minx = int(max(0, x-RADIUS))
			maxx = int(min(x+RADIUS, img.shape[1]-1))
			miny = int(max(0, y-RADIUS))
			maxy = int(min(y+RADIUS, img.shape[0]-1))
			# if i == 159:
			# print(x)
			# print(y)
			# print(minx)
			# print(maxx)
			# print(miny)
			# print(maxy)
			img[miny:maxy, minx:maxx, 0] = 255
			img[miny:maxy, minx:maxx, 1] = 0
			img[miny:maxy, minx:maxx, 2] = 0
		# video_file.write(img)
	# cv2.destroyAllWindows()
	# video_file.release()
		# sci.imsave(img_dst_path, img)
		cv2.imwrite(img_dst_path, img)
	# print(gaze_file.readline())
	gaze_file.close()
	# break


# generate video from speech and gaze images
