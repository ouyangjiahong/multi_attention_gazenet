# make dataset from npz
# put this file outside data_collection folder

from scipy import misc as sci
import numpy as np
import os
import math
from shutil import copyfile

data_dir = 'annotation_dataset/'
dst_data_dir = 'gaze_dataset/'

# load npz data
file = np.load('annotation.npz')
name_list = file['arr_0']
target_list = file['arr_1']
start_list = file['arr_2']
end_list = file['arr_3']
hand_list = file['arr_4']
print(len(target_list))

# make datasets
for i in range(1,5):
	directory = os.path.dirname(dst_data_dir + str(i) + '/')
	if not os.path.exists(directory):
		os.makedirs(directory)

num = name_list.shape[0]
valid_period = []
for i in range(num):
	# print(i)
	name = name_list[i]
	print(name)
	target = target_list[i]
	# print('target')
	# print(target)
	start_count = int(start_list[i])
	end_count = int(end_list[i])
	valid_period.append([start_count, end_count])
	#two valid period per interaction
	if i != num - 1 and name == name_list[i+1]:
		continue
	hand_count = int(hand_list[i])

	# make a new subdir
	img_dst_dir = dst_data_dir + str(target) + '/' + name + '/'
	if not os.path.exists(img_dst_dir):
		os.makedirs(img_dst_dir)

	# save valid images
	print(valid_period)
	num_img = hand_count - 2
	for j in range(1, hand_count):
		count = str("%06d" % j)
		img_name = data_dir + name + '/' + name + count + '.jpg'
		img_dst_name =  img_dst_dir + count + '.jpg'
		# print(img_name)
		# no input image
		if not os.path.exists(img_name):
			num_img -= 1
			print("miss")
			continue
		copyfile(img_name, img_dst_name)
		# print(img_name)
	print("number of img")
	print(num_img)

	# make the label
	label = np.zeros((hand_count-2,1))
	for p in valid_period:
		label[p[0]-1:p[1]-1] = target
	print('length of label')
	print(len(label))
	np.save(img_dst_dir + 'label.npy', label)

	# save valid gaze txt
	gaze_src_path = data_dir + name + '/' + name + 'testfile.txt'
	gaze_src_file = open(gaze_src_path)
	gaze_list_ori = gaze_src_file.readlines()
	gaze_src_file.close()
	gaze_list = gaze_list_ori[0:3*(hand_count-2)]
	gaze_list = [float(gaze_list[k]) for k in range(len(gaze_list))]
	gaze_list = np.reshape(np.array(gaze_list), (-1, 3))
	print(gaze_list.shape)

	print('length of gaze list')
	print(len(gaze_list)//3)
	# print(len(gaze_list))
	np.save(img_dst_dir + 'gaze.npy', gaze_list)

	valid_period = []
	# print('clear')
