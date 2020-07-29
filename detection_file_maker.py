import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from yolo.yolov3 import Create_Yolov3
from yolo.utils import image_preprocess, postprocess_boxes, nms, read_class_names 
from yolo.configs import *

annot_path = './model_data/pascal_voc07_test.txt'
CURRENT_KNOWN_OBJECTS = EVALUATION_CLASSES = './model_data/classes_to_evaluate.txt'
PASCAL_VOC_ALL_CLASSES = './model_data/pascal_voc07_cls_names.txt'
dt = './dt/'


iou_threshold = 0.5 
score_threshold = 0.3 


input_size = YOLO_INPUT_SIZE
CURRENT_KNOWN_OBJECTS_NAME = read_class_names(EVALUATION_CLASSES,dot_name_file= False)
PASCAL_VOC_ALL_CLASSES_NAME = read_class_names(PASCAL_VOC_ALL_CLASSES)


yolo = Create_Yolov3(input_size=input_size,CLASSES=CURRENT_KNOWN_OBJECTS)
yolo.load_weights('./checkpoints/yolov3_custom_val_loss_ 808.03')


def load_annotations(annot_path):
		final_annotations = []
		with open(annot_path, 'r') as f:
			txt = f.readlines()
			annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
		np.random.shuffle(annotations)
		
		for annotation in annotations:
			# fully parse annotations
			line = annotation.split()
			image_path, index = "", 1
			for i, one_line in enumerate(line):
				if not one_line.replace(",","").isnumeric():
					if image_path != "": image_path += " "
					image_path += one_line
				else:
					index = i
					break
			if not os.path.exists(image_path):
				raise KeyError("%s does not exist ... " %image_path)
			if TRAIN_LOAD_IMAGES_TO_RAM: image_path = cv2.imread(image_path)
			final_annotations.append([image_path, line[index:]])
			

		return final_annotations



final_annotations = load_annotations(annot_path)

for annotation in final_annotations:
	image_path = annotation[0]
	image = cv2.imread(image_path)
	original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	image = image_preprocess(original_image, (416,416), gt_boxes=None)

	
	image_exp = np.expand_dims(image, 0)



	pred_bbox = yolo.predict(image_exp)
	
	pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
	pred_bbox = tf.concat(pred_bbox, axis=0)

	bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
	bboxes = nms(bboxes, iou_threshold, method='nms')
	

	bboxes_true = np.array([list(map(int, box.split(','))) for box in annotation[1]])
	contains_object_desired = False
	for bbox_true in bboxes_true:
		class_name_from_pascal = PASCAL_VOC_ALL_CLASSES_NAME[bbox_true[4]] 
		if class_name_from_pascal in CURRENT_KNOWN_OBJECTS_NAME:
			contains_object_desired = True 
	if contains_object_desired:
		image_path =annotation[0]
		file_name = image_path.split('/')[-1].split('.')[0]+'.txt'
		file_object = open(dt+file_name,'w')
		if bboxes != []: 	
			for bbox in bboxes:
				my_string =CURRENT_KNOWN_OBJECTS_NAME[int(bbox[-1])] +' '+\
					str(bbox[4])+' '+str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2])+' '+str(bbox[3])
				file_object.writelines(my_string+'\n')
			file_object.close()
		else:
			file_object.close() 
	print('Please wait...')


		





