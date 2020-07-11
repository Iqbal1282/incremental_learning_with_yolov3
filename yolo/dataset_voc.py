
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from yolo.utils import read_class_names, image_preprocess
from yolo.yolov3 import bbox_iou
from yolo.configs import *


PASCAL_VOC_ALL_CLASSES = './model_data/pascal_voc07_cls_names.txt'


class Dataset(object):
    # Dataset preprocess implementation
    def __init__(self, dataset_type, NEW_CLASSES_TO_LEARN, TOTAL_CLASSES_WILL_KNOW_AFTER_THIS):
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        self.input_sizes = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug    = TRAIN_DATA_AUG   if dataset_type == 'train' else TEST_DATA_AUG

        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)

        self.classes = read_class_names(TOTAL_CLASSES_WILL_KNOW_AFTER_THIS, dot_name_file= False)
        self.num_classes = len(self.classes)

        self.anchors = (np.array(YOLO_ANCHORS).T/self.strides).T
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.TOTAL_CLASSES_WILL_KNOW_AFTER_THIS = TOTAL_CLASSES_WILL_KNOW_AFTER_THIS


        self.new_classes = self.index_mapping_file(PASCAL_VOC_ALL_CLASSES, NEW_CLASSES_TO_LEARN) # [3,8]

        self.count_image_goes = 0


    def index_mapping_file(self,larger_one, smaller_one):
        indexes = []
        items_l = read_class_names(larger_one, dot_name_file= False)
        items_s = read_class_names(smaller_one, dot_name_file= False)
        
        for item in items_s:
            indexes.append(items_l.index(item))
        return indexes
    def index_mapping_array(self, items_l, items_s):
        indexes = []

        for item in items_s:
            indexes.append(items_l.index(item))
        return indexes




    def load_annotations(self, dataset_type):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
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

    def __iter__(self):
        return self

    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice([self.train_input_sizes])
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4), dtype=np.float32)
            
            

            if self.count_image_goes < self.num_samples:
                num = 0 
                while num < self.batch_size and self.count_image_goes <self.num_samples:

                    annotation = self.annotations[self.count_image_goes]
                    self.count_image_goes +=1  
                    image, bboxes = self.parse_annotation(annotation)

                    ########################
                    contains_desired_object= False
                    for bbox in bboxes:
                        if bbox[4] in self.new_classes:
                            contains_desired_object = True 
                    # if contains_desired_object:
                    #     for bbox in bboxes:
                    #         x1,y1, x2, y2 = bbox[0],bbox[1], bbox[2], bbox[3]
                    #         cv2.rectangle(image, (x1,y1),(x2,y2), (255,0,255), 2)

                    #     cv2.imshow('image', image)
                    #     cv2.waitKey(0)
                    #     cv2.destroyAllWindows()


                    if contains_desired_object:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)
                        #######################################
                        
                        batch_image[num, :, :, :] = image
                        batch_label_sbbox[num, :, :, :, :] = label_sbbox
                        batch_label_mbbox[num, :, :, :, :] = label_mbbox
                        batch_label_lbbox[num, :, :, :, :] = label_lbbox
                        batch_sbboxes[num, :, :] = sbboxes
                        batch_mbboxes[num, :, :] = mbboxes
                        batch_lbboxes[num, :, :] = lbboxes
                        #img_count+=1
                        num +=1

                
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)
            else:
                self.batch_count = 0
                self.count_image_goes = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def parse_annotation(self, annotation):
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image = annotation[0]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)
            
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        image, bboxes = image_preprocess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        EXPANDED_CLASSES_NAME = read_class_names(self.TOTAL_CLASSES_WILL_KNOW_AFTER_THIS, dot_name_file= False)
        VOC_CLASSES_NAME = list(read_class_names(PASCAL_VOC_ALL_CLASSES, dot_name_file= False))
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]
        bbox_count = np.zeros((3,))
 
        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4] 
            if bbox_class_ind in self.new_classes: 
                onehot = np.zeros(self.num_classes, dtype=np.float)
                bbox_class_ind_new =EXPANDED_CLASSES_NAME.index(VOC_CLASSES_NAME[bbox_class_ind])
                
                onehot[bbox_class_ind_new] = 1.0
                uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
                deta = 0.01
                smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

                bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
                bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

                iou = []
                exist_positive = False
                for i in range(3):
                    anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                    anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                    anchors_xywh[:, 2:4] = self.anchors[i]

                    iou_scale = bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)
                    iou.append(iou_scale)
                    iou_mask = iou_scale > 0.3

                    if np.any(iou_mask):
                        xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                        label[i][yind, xind, iou_mask, :] = 0
                        label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                        label[i][yind, xind, iou_mask, 4:5] = 1.0
                        label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                        bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                        bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                        bbox_count[i] += 1

                        exist_positive = True

                if not exist_positive:
                    best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                    best_detect = int(best_anchor_ind / self.anchor_per_scale)
                    best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                    xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                    label[best_detect][yind, xind, best_anchor, :] = 0
                    label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                    label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                    label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                    bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                    bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        raise StopIteration
        return self.num_batchs








