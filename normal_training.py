import os
import argparse 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import shutil
import numpy as np
import tensorflow as tf
#from tensorflow.keras.utils import plot_model
from yolo.dataset_voc import Dataset
from yolo.yolov3 import Create_Yolov3, YOLOv3, decode, compute_loss
from yolo.configs import *

#### for dataset 
NEW_CLASSES_TO_LEARN = './model_data/new_class_to_learn.txt'
MODEL_OBJECTS_BEFORE_INCRE = './model_data/model_objects_before_incre.txt'
MODEL_OBJECTS_AFTER_INCRE = './model_data/model_objects_after_incre.txt'
PASCAL_VOC_ALL_CLASSES = './model_data/pascal_voc07_cls_names.txt'


def main():
	global TRAIN_FROM_CHECKPOINT
	
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if len(gpus) > 0:
		try: tf.config.experimental.set_memory_growth(gpus[0], True)
		except RuntimeError: pass

	# if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
	# writer = tf.summary.create_file_writer(TRAIN_LOGDIR)


	trainset = Dataset('train', NEW_CLASSES_TO_LEARN, MODEL_OBJECTS_AFTER_INCRE)
	testset = Dataset('test', NEW_CLASSES_TO_LEARN, MODEL_OBJECTS_AFTER_INCRE)
	
	steps_per_epoch =0
	for _  in trainset:
		steps_per_epoch+=1 

	print(steps_per_epoch)

	global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
	warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
	total_steps = TRAIN_EPOCHS * steps_per_epoch

	#yolo_prev = Create_Yolov3(input_size=YOLO_INPUT_SIZE,training= False, CLASSES=MODEL_OBJECTS_BEFORE_INCRE)
	yolo_new = Create_Yolov3(input_size=YOLO_INPUT_SIZE, training= True, 
		CLASSES=MODEL_OBJECTS_AFTER_INCRE, dot_name_file = False)


	
	# # load the weights for the previous checkpont 
	#yolo_new.load_weights('./checkpoints/yolov3_custom_val_loss_  23.96')
	# load_yolo_weights(yolo_prev, weights_file)
	yolo_new.load_weights('./checkpoints/yolov3_custom_val_loss_  12.90')

	# for i, l in enumerate(yolo_prev.layers):
	# 	layer_weights = l.get_weights()
	# 	if layer_weights != []:
	# 		try:
	# 			#print(layer_weights)
	# 			yolo_new.layers[i].set_weights(layer_weights)
	# 			#print(yolo_new.layers[i].get_weights())
	# 		except:
	# 			print('skiping', yolo_prev.layers[i].name)




	optimizer = tf.keras.optimizers.Adam(lr=0.00001)

	def train_step(image_data, target):
		with tf.GradientTape() as tape:
			pred_result = yolo_new(image_data, training=True)
			giou_loss=conf_loss=prob_loss=0

			# optimizing process
			grid = 3 
			for i in range(grid):
				conv, pred = pred_result[i*2], pred_result[i*2+1]
				loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=MODEL_OBJECTS_AFTER_INCRE)
				giou_loss += loss_items[0]
				conf_loss += loss_items[1]
				prob_loss += loss_items[2]

			total_loss = giou_loss + conf_loss + prob_loss

			gradients = tape.gradient(total_loss, yolo_new.trainable_variables)
			optimizer.apply_gradients(zip(gradients, yolo_new.trainable_variables))

			# update learning rate
			# about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
			global_steps.assign_add(1)
		# if global_steps < warmup_steps:# and not TRAIN_TRANSFER:
		# 	lr = global_steps / warmup_steps * TRAIN_LR_INIT
		# else:
		# 	lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
		# 			(1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))
		# 	optimizer.lr.assign(lr.numpy())

			# writing summary data
			# with writer.as_default():
			#     tf.summary.scalar("lr", optimizer.lr, step=global_steps)
			#     tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
			#     tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
			#     tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
			#     tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
			# writer.flush()


			
		return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

	#validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
	def validate_step(image_data, target):
		#with tf.GradientTape() as tape:
		pred_result = yolo_new(image_data, training=False)
		giou_loss=conf_loss=prob_loss=0
		print('Validation in Process wait.... ')

		# optimizing process
		grid = 3 
		for i in range(grid):
			conv, pred = pred_result[i*2], pred_result[i*2+1]
			loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=MODEL_OBJECTS_AFTER_INCRE)
			giou_loss += loss_items[0]
			conf_loss += loss_items[1]
			prob_loss += loss_items[2]

		total_loss = giou_loss + conf_loss + prob_loss
			
		return giou_loss.numpy(), conf_loss.numpy(), prob_loss.numpy(), total_loss.numpy()


	best_val_loss = 1000 # should be large at start
	for epoch in range(TRAIN_EPOCHS):
		for image_data, target in trainset:
			results = train_step(image_data, target)
			cur_step = results[0]%steps_per_epoch
			print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, giou_loss:{:7.2f}, conf_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
				  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5]))


		
		count, giou_val, conf_val, prob_val, total_val = 0., 0, 0, 0, 0
		for image_data, target in testset:
			results = validate_step(image_data, target)
			count += 1
			giou_val += results[0]
			conf_val += results[1]
			prob_val += results[2]
			total_val += results[3]
		# writing validate summary data
		# with validate_writer.as_default():
		#     tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
		#     tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
		#     tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
		#     tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
		# validate_writer.flush()
			
		print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
			  format(giou_val/count, conf_val/count, prob_val/count, total_val/count))

		yolo_new.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"_val_loss_{:7.2f}".format(total_val/count)))




main()













