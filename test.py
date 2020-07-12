import os
import argparse 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import numpy as np
import tensorflow as tf

from yolo.dataset_voc import Dataset
from yolo.yolov3 import Create_Yolov3, YOLOv3, decode, compute_loss, bbox_iou
from yolo.configs import *
from yolo.utils import read_class_names, nms, postprocess_boxes 
from collections import Counter


NEW_CLASSES_TO_LEARN = './model_data/new_class_to_learn.txt'
MODEL_OBJECTS_BEFORE_INCRE = './model_data/model_objects_before_incre.txt'
MODEL_OBJECTS_AFTER_INCRE = './model_data/model_objects_after_incre.txt'
PASCAL_VOC_ALL_CLASSES = './model_data/pascal_voc07_cls_names.txt'


TEST_CLASSES = MODEL_OBJECTS_AFTER_INCRE
#trainset = Dataset('train', NEW_CLASSES_TO_LEARN, MODEL_OBJECTS_AFTER_INCRE)
testset = Dataset('test', TEST_CLASSES, TEST_CLASSES)

steps_per_epoch =0
for _  in testset:
	steps_per_epoch+=1 

print(steps_per_epoch)


yolo = Create_Yolov3(input_size=YOLO_INPUT_SIZE, training= False, 
		CLASSES=MODEL_OBJECTS_AFTER_INCRE, dot_name_file = False)

yolo.load_weights('./checkpoint_for_increment_cat/yolov3_custom_val_loss_38778.18')

num_classes = len(read_class_names(MODEL_OBJECTS_AFTER_INCRE))






def evaluate(y_pred, y_true_temp, num_classes, score_thresh=0.4, iou_thresh=0.5):
	y_true = [y_true_temp[0][0], y_true_temp[1][0], y_true_temp[2][0]]

	num_images = y_true[0].shape[0]
	true_labels_dict   = {i:0 for i in range(num_classes)} # {class: count}
	pred_labels_dict   = {i:0 for i in range(num_classes)}
	true_positive_dict = {i:0 for i in range(num_classes)}

	for i in range(num_images):
		true_labels_list, true_boxes_list = [], []
		for j in range(3): # three feature maps
			true_probs_temp = y_true[j][i][...,5: ]
			true_boxes_temp = y_true[j][i][...,0:4]



			object_mask = true_probs_temp.sum(axis=-1) > 0
			#print(true_probs_temp.sum(axis=-1))


			true_probs_temp = true_probs_temp[object_mask]
			true_boxes_temp = true_boxes_temp[object_mask]

			# print(true_probs_temp.shape) # [7,3]
			# print(true_boxes_temp.shape) # [7,4]



			true_labels_list += np.argmax(true_probs_temp, axis=-1).tolist()
			true_boxes_list  += true_boxes_temp.tolist()
			

		if len(true_labels_list) != 0:
			for cls, count in Counter(true_labels_list).items(): 
				true_labels_dict[cls] += count
			#print(true_labels_dict)

		pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in y_pred]
		pred_bbox = tf.concat(pred_bbox, axis = 0)

		#print(pred_bbox.shape)
		#bboxes = pred_bbox
		bboxes = postprocess_boxes(pred_bbox, np.zeros((416,416,3)), 416, score_thresh)
		best_bboxes = nms(bboxes, iou_thresh, sigma=0.3, method='nms')
		# print(best_bboxes[0].shape)
		# print(best_bboxes)
		# raise StopIteration
		# nms(bboxes, iou_threshold, sigma=0.3, method='nms')
		#:param bboxes: (xmin, ymin, xmax, ymax, score, class)

		# pred_boxes, pred_confs, pred_labels = cpu_nms(pred_boxes, pred_confs*pred_probs, num_classes,
		# 											  score_thresh=score_thresh, iou_thresh=iou_thresh)

		pred_boxes = []
		pred_labels = []
		pred_confs = []

		for bbox in bboxes:
			pred_boxes.append(bbox[:4])
			pred_labels.append(bbox[5:])
			pred_confs.append(bbox[4:5])

		if len(true_labels_list) == 0:
			continue

		true_boxes = np.array(true_boxes_list)
		box_centers, box_sizes = true_boxes[:,0:2], true_boxes[:,2:4]

		true_boxes[:,0:2] = box_centers - box_sizes / 2.
		true_boxes[:,2:4] = true_boxes[:,0:2] + box_sizes



		pred_labels_list = [] if pred_labels is [] else pred_labels
		if pred_labels_list == []: continue

		detected = []

		for k in range(len(true_labels_list)):
			# compute iou between predicted box and ground_truth boxes

			iou = bbox_iou(np.array(true_boxes[k:k+1]), np.array(pred_boxes))
			m = np.argmax(iou) # Extract index of largest overlap
			if iou[m] >= iou_thresh and true_labels_list[k] == pred_labels_list[m] and m not in detected:
				pred_labels_dict[true_labels_list[k]] += 1
				detected.append(m)
		pred_labels_list = [pred_labels_list[m] for m in detected]

		for c in range(num_classes):
			t = true_labels_list.count(c)
			p = pred_labels_list.count(c)
			true_positive_dict[c] += p if t >= p else t

	recall    = sum(true_positive_dict.values()) / (sum(true_labels_dict.values()) + 1e-6)
	precision = sum(true_positive_dict.values()) / (sum(pred_labels_dict.values()) + 1e-6)
	avg_prec  = [true_positive_dict[i] / (true_labels_dict[i] + 1e-6) for i in range(num_classes)]
	mAP       = sum(avg_prec) / (sum([avg_prec[i] != 0 for i in range(num_classes)]) + 1e-6)

	return recall, precision, mAP



for i in range(1):
	t_recall=t_precision= t_mAP= 0.  
	count = 0
	for image_batch, true_label_batch in testset:
		pred = yolo.predict(image_batch)
		recall, precision, mAP = evaluate(pred, true_label_batch, num_classes, score_thresh = 0.4, iou_thresh = 0.5)
		t_recall += recall
		t_precision += precision 
		t_mAP += mAP
		count +=1 

		print('Step :{:7d}/{},recall :{:7.3f}, precision : {:7.3f}, mAP : {:7.3f}'.format(count,steps_per_epoch, recall, precision, mAP))
		#print(recall, precision, mAP)
		print(t_recall, t_precision, t_mAP, count)


	print('average_recall :{:7.3f}, average_precision : {:7.3f}, average_mAP : {:7.3f}'.format(t_recall/count, 
		t_precision/count, t_mAP/count))


