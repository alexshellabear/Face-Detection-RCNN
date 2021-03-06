import cv2
import keras
import argparse
import os
import random
import pickle

"""
    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Purpose:
        Create a labelled training set of data from the selective search from CV2

    Lessons Learnt
        1) Step by step guide for R-CNN
            https://towardsdatascience.com/step-by-step-r-cnn-implementation-from-scratch-in-python-e97101ccde55
        2) OpenCV has selective search as part of the extended image preprocessing
            https://docs.opencv.org/master/df/d2d/group__ximgproc.html
        3) Loop through valid file extensions
            https://stackoverflow.com/questions/31948528/python-loop-through-files-of-certain-extensions   
        4) Understanding selective search algorithms that are present in cv2
            https://learnopencv.com/selective-search-for-object-detection-cpp-python/ 
        5) First set of training didn't go as well, most likely because the negative images chosen were not representative of the sample being chosen.
            Using the precicted samples from the first iteration of training so I could determine what the model was. Hence the dataset needs to be more representative
            proposed features
                Scale
        6) When labeling my face for the first time I was selecting other objects and grouping them such as my hair & hat.
            Removing this made it easier for the selective search to find occurances of my face

"""
config = {
    "ImagePath" : "1. Data Gen\\1. Data\\WIN_20210216_09_48_49_Pro.jpg"
    ,"ClassifierInput": (224,224)
    ,"PathToData" : "1. Data Gen\\1. Data"
    ,"ValidImageFileExtensions" : [".jpg",".png"]
    ,"ValidLabelFileExtensions" : ['.csv']
    ,"Model": {
        "iouThreshhold" : 0.75
    }
}

def get_iou(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_list_of_data_and_labels():
    """
        Loop through data and find image files & corresponding csv files. For those that do not have corresponding file extensions remove from list

        Assumptions
            1) Those images which have a ground truth will have the same file name but with a csv file extension
            2) Length of the image list and label list should be the same
            
    """
    list_of_imgs = []
    list_of_img_labels = []
    for root, dirs, files in os.walk(config["PathToData"], topdown=False):
        for f in files:
            ext = os.path.splitext(f)[-1].lower()

            if ext in config["ValidImageFileExtensions"]:
                list_of_imgs.append(os.path.join(root, f))
            if ext in config["ValidLabelFileExtensions"]:
                list_of_img_labels.append(os.path.join(root, f))

    list_of_imgs_with_labels = []
    list_of_corresponing_labels = []
    for img_full_file_name in list_of_imgs:
        img_file_name = os.path.splitext(img_full_file_name)[0].lower()
        corresponding_label = [label_full_file_name for label_full_file_name in list_of_img_labels if os.path.splitext(label_full_file_name)[0].lower() == img_file_name]
        if len(corresponding_label) != 0:
            list_of_imgs_with_labels.append(img_full_file_name)
            list_of_corresponing_labels.append(corresponding_label[0])

    assert len(list_of_imgs_with_labels) == len(list_of_corresponing_labels)

    return list_of_imgs_with_labels, list_of_corresponing_labels 

def read_img_labels(img_label_path):
    with open(img_label_path,"r") as img_label_file:
        data_lines = img_label_file.readlines()
        x1,y1,x2,y2 = data_lines[1].split("\t")
        img_label_file.close()

    ground_truth_bounding_box = {
        "x1" : int(x1)
        ,"y1" : int(y1)
        ,"x2" : int(x2)
        ,"y2" : int(y2)
    }
    return ground_truth_bounding_box

def get_rectangles_of_interest_using_selective_serach(base_image):
    """
        Description: Uses cv2's inbuilt fast selective search method to return potential rectangle of interest
    """
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(base_image)
    selective_search.switchToSelectiveSearchFast() # lower likelihood of getting correct bounding boxes but quicker
    potential_rectangles_of_interest = selective_search.process()
    return potential_rectangles_of_interest

def create_more_foreground_samples_if_not_enough_exist(base_image,img_path,ground_truth_bounding_box,labels,min_num_foregrounds=50):
    """
        Purpose: Generates foreground labels from ground truth if there are not the minimum number of foreground labels generated from selective search method
    """
    foreground_labelled_imgs = len([v for v in labels if v["Label"] == "Foreground"])

    foreground_labels_generated_from_ground_truth = []
    if foreground_labelled_imgs < min_num_foregrounds: # Artificially boost the number of positive images.
        max_iou_found = max([v['IOU'] for v in labels])
        print(f"Not enough foreground images detected, only {len([v for v in labels if v['Label'] == 'Foreground'])} found, max iou = {max_iou_found}  create from ground truth. Generate another {min_num_foregrounds - foreground_labelled_imgs} images")
        for j in range(min_num_foregrounds - foreground_labelled_imgs):
            foreground_labels_generated_from_ground_truth.append(generate_box_from_ground_truth(base_image,ground_truth_bounding_box,img_path))
    return foreground_labels_generated_from_ground_truth

def generate_labeled_data(img_path:str,img_label_path:str):
    """
        Purpose: Labels data into foreground (E.g. my face) and background (everything else).
                 cv2's inbuilt Selective search is used to generate images, most will be background.
                 A minimum number of foreground labels are specified, more foreground images are generated from ground truth and added to the set.
    """
    base_image = cv2.imread(img_path)
    ground_truth_bounding_box = read_img_labels(img_label_path)
    rects = get_rectangles_of_interest_using_selective_serach(base_image)

    selective_search_labels = []
    for j, box in enumerate(rects):
        x,y,w,h = box
        cropped_resized_img = cv2.resize(base_image[y:y+h,x:x+w], config["ClassifierInput"], interpolation = cv2.INTER_AREA)

        selected_search_box = {
            "x1" : x
            ,"y1" : y
            ,"x2" : x+w
            ,"y2" : y+w
        }
        
        iou = get_iou(ground_truth_bounding_box, selected_search_box)
        if iou > config["Model"]["iouThreshhold"]: # You want this to be zero so there is no overlap in between yes and no
            selective_search_labels.append({
                "ImagePath" : img_path
                ,"Label" :"Foreground"
                ,"IOU" : iou
                ,"Box" : selected_search_box
                })
        elif iou == 0: # You want this to be zero so there is no overlap in between yes and no
            selective_search_labels.append({
                "ImagePath" : img_path
                ,"Label" :"Background"
                ,"IOU" : iou 
                ,"Box" : selected_search_box
                })
            
    labels = selective_search_labels

    foreground_labels_generated_from_ground_truth = create_more_foreground_samples_if_not_enough_exist(base_image,img_path,ground_truth_bounding_box,labels)
    for foreground_gen_label in foreground_labels_generated_from_ground_truth:
        labels.append(foreground_gen_label)

    print(f"img[{img_path}] generated {len(labels)} labels")
    return labels

def generate_box_from_ground_truth(base_image,ground_truth_bounding_box,img_path,max_measurement_diff=0.1,min_iou=0.9):
    """
        purpose:
            Generate correct labels from ground truth given the min IOU (Intersection over Union) and max difference in width and height.

        Return:
            Bounding box meeting criteria and the iou
    """
    img_height, img_width, _ =  base_image.shape
    box_height = ground_truth_bounding_box["y2"]-ground_truth_bounding_box["y1"]
    box_width = ground_truth_bounding_box["x2"]-ground_truth_bounding_box["x1"]

    while(1):
        try:
            x_movement = round(random.randrange(-100,100)/100*max_measurement_diff*box_width)
            y_movement = round(random.randrange(-100,100)/100*max_measurement_diff*box_height)
            new_label_bounding_box = {
                "x1" : ground_truth_bounding_box["x1"] + x_movement
                ,"x2" : ground_truth_bounding_box["x2"] + x_movement
                ,"y1" : ground_truth_bounding_box["y1"] + y_movement
                ,"y2" : ground_truth_bounding_box["y2"] + y_movement
            }

            # Error Check
            assert new_label_bounding_box["x1"] >= 0
            assert new_label_bounding_box["y1"] >= 0
            assert new_label_bounding_box["x2"] <= img_width
            assert new_label_bounding_box["y2"] <= img_height

            iou = get_iou(ground_truth_bounding_box,new_label_bounding_box)
            assert iou >= min_iou

            return {
                "ImagePath" : img_path
                ,"Label" :"Foreground"
                ,"IOU" : iou 
                ,"Box" : new_label_bounding_box
                }
        except:
            pass

def get_area_of_box(box: dict):
    """
        Purpose: Gets the area of the box
        Assumption 1: x1 < x2 & y1 < y2
    """
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    area = width * height
    return area

if __name__ == "__main__":
    print("starting...")
    list_of_imgs, list_of_img_labels = get_list_of_data_and_labels()

    labels_list = [generate_labeled_data(img_path,list_of_img_labels[i]) for i, img_path in enumerate(list_of_imgs)]
    
    foreground_labels = [individual_box for img_results in labels_list for individual_box in img_results if individual_box["Label"] == "Foreground"]
    background_labels = [individual_box for img_results in labels_list for individual_box in img_results if individual_box["Label"] == "Background"]
    
    pickle.dump(foreground_labels,open( config["PathToData"]+os.sep+"foreground_labels.p", "wb" ))
    pickle.dump(background_labels,open( config["PathToData"]+os.sep+"background_labels.p", "wb" ))

    
    print("finishing...")