import numpy as np 
import os 
TRAIN_LOAD_IMAGES_TO_RAM = False 
import cv2 

PASCAL_VOC_ALL_CLASSES = './model_data/pascal_voc07_cls_names.txt'
EVALUATION_CLASSES = './model_data/classes_to_evaluate.txt'
gt = './gt/'



def read_class_names(class_file_name,dot_name_file= False):
    # loads class name from a file
    if dot_name_file:
        names = {}
        with open(class_file_name, 'r') as data:
            for ID, name in enumerate(data):
                names[ID] = name.strip('\n')
        return names
    else:
        names = []
        with open(class_file_name,'r') as file:
            lines = file.readlines()
            for line in lines:
                names.append(line.strip('\n'))
        return names 


PASCAL_VOC_ALL_CLASSES_NAME = read_class_names(PASCAL_VOC_ALL_CLASSES)
EVALUATION_CLASSES_NAME = read_class_names(EVALUATION_CLASSES)


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



def parse_annotation(annotations): 
    for annotation in annotations:
        image = annotation[0]          
        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])
        file_name = image.split('/')[-1].split('.')[0]+'.txt'
        bboxes_true = np.array([list(map(int, box.split(','))) for box in annotation[1]])
        contains_object_desired = False
        for bbox_true in bboxes_true:
            class_name_from_pascal = PASCAL_VOC_ALL_CLASSES_NAME[bbox_true[4]] 
            if class_name_from_pascal in EVALUATION_CLASSES_NAME:
                contains_object_desired = True 
        if contains_object_desired:
            image_path =annotation[0]
            file_name = image_path.split('/')[-1].split('.')[0]+'.txt'
            file_object = open(gt+file_name,'w')
            for bbox in bboxes:
                label= bbox[-1]
                label_name = PASCAL_VOC_ALL_CLASSES_NAME[label]
                if label_name not in EVALUATION_CLASSES_NAME:
                    continue
                boundary_bbox = bbox[:4]
                file_object.writelines(label_name+' '+str(boundary_bbox[0])+' '+\
                    str(boundary_bbox[1])+' '+str(boundary_bbox[2])+' '+str(boundary_bbox[3])+'\n')

            file_object.close()



annot_path = './model_data/pascal_voc07_test.txt'

final_annotations = load_annotations(annot_path)
parse_annotation(final_annotations)
#print(final_annotations)
# print(image, bboxes)

# print(image.split('/')[-1].split('.')[0]+'.txt')