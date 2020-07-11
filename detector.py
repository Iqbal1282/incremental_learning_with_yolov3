import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolo.yolov3 import Create_Yolov3
from yolo.utils import load_yolo_weights, detect_image, detect_realtime ,detect_video
from yolo.configs import *

input_size = YOLO_INPUT_SIZE

image_path   = "./sample_data/car_person.jpg"
video_path = "./sample_data/sample.mp4"
#video_path   = "./Images/horse.mp4"

CURRENT_KNOWN_OBJECTS = './model_data/current_objects.txt'

yolo = Create_Yolov3(input_size=input_size,CLASSES=CURRENT_KNOWN_OBJECTS)
#load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
yolo.load_weights('./checkpoints/yolov3_custom_val_loss_  12.90')

# detect_image(yolo, image_path, 'output1.jpg', input_size=input_size, show=True, rectangle_colors='',
# 	CLASSES=CURRENT_KNOWN_OBJECTS )
# detect_video(yolo, video_path, 'dist7.mp4', input_size=input_size, show=True, rectangle_colors='',
# 	CLASSES=CURRENT_KNOWN_OBJECTS)
# detect_realtime(yolo, '', input_size=input_size, show=True, rectangle_colors=(255, 0, 0), 
# 	CLASSES = CURRENT_KNOWN_OBJECTS)



if __name__ = '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--type')
	args = parser.parse_args()

	if args.type == 'video':
		detect_video(yolo, video_path, 'output.mp4', input_size=input_size, show=True, rectangle_colors='',
			CLASSES=CURRENT_KNOWN_OBJECTS)

	elif args.type == 'image':

		detect_image(yolo, image_path, 'output.jpg', input_size=input_size, show=True, rectangle_colors='',
			CLASSES=CURRENT_KNOWN_OBJECTS )

	else:
		raise ValueError 



