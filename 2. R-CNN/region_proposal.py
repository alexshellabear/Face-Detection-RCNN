"""
    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Research
    1) Step by step explanation
        https://dongjk.github.io/code/object+detection/keras/2018/05/21/Faster_R-CNN_step_by_step,_Part_I.html
    2) vgg with top=false will only output the feature maps which is (7,7,512), other solutions will have different features produced
        https://github.com/keras-team/keras/issues/4465
    3) Understanding anchor boxes
        https://machinelearningmastery.com/padding-and-stride-for-convolutional-neural-networks/
    4) Faster RCNN - how they calculate stride
        https://stats.stackexchange.com/questions/314823/how-is-the-stride-calculated-in-the-faster-rcnn-paper
    5) Good article on Faster RCNN explained, can only access via print to pdf
        https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8
    6) Indicating that Anchor boxes should be determine by ratio and scale
        ratio should be width:height of 1:2 1:1 2:1
        scale should be 1 1/2 1/3
        https://keras.io/examples/vision/retinanet/
    7) Best explanation of anchor boxes
        https://www.mathworks.com/help/vision/ug/anchor-boxes-for-object-detection.html#:~:text=Anchor%20boxes%20are%20a%20set,sizes%20in%20your%20training%20datasets.
    8) Summary of object detection history, interesting read
        https://dudeperf3ct.github.io/object/detection/2019/01/07/Mystery-of-Object-Detection/
    9) What to do if there are no good potential bounding boxes have a high enough IOU.
       Suggestion is to just get the next best option.
       https://github.com/you359/Keras-FasterRCNN/blob/eb67ad5d946581344f614faa1e3ee7902f429ce3/keras_frcnn/data_generators.py#L203 
    10) Exactly what I was wanting on RPN - have to use pdf
        https://towardsdatascience.com/building-a-mask-r-cnn-from-scratch-in-tensorflow-and-keras-c49c72acc272
    11) Custom loss functions in Keras
        https://stackoverflow.com/questions/54977311/what-is-loss-cls-and-loss-bbox-and-why-are-they-always-zero-in-training
    12) Line 5, tells what the bounding box format should be before passing it to the neural net
        Must be a numpy array of [x_centre,y_centre,width,height]
        Normalise all values using a np.std
        https://github.com/jrieke/shape-detection/blob/master/single-rectangle.ipynb

    Lessons Learnt
        1) Do not exceed the length of the image with the anchor boxes
        2) np.array.reshape() allows you to change the change of your array, in this case you can flatten your array using
            np.array.reshape(-1), this will let np choose the shape as long as it all adds up to np.array.shape
            np.array.reshape(-1,1) means the array will be flattened but each value will have it's own cell
            https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape
        3) Array slicing with numpy or lists
            list[start:stop:step]
            list[:] = all
            list[0::2] = start at 0 and only return every 2nd item
        4) If you want to explicitely add a new axis with the list then you can use
            np.newaxis which explicitely adds a new axis
            only works for numpy arrays not lists
            https://stackoverflow.com/questions/29241056/how-does-numpy-newaxis-work-and-when-to-use-it
        5) using bounding box regression means that the output of the RPN is in the following
            deltas (slight movements) = [dx_point_c,dy_point_c,dw_in_log_space,dh_in_log_space]
            It is interesting that regression for width and height are in the log space
            https://datascience.stackexchange.com/questions/30557/how-does-the-bounding-box-regressor-work-in-fast-r-cnn
        6) Originally was really confused as to why the output of the Region Proposal Network RPN was not giving the number
            of rows output was different to the number of proposed bounding boxes. I accidentally passed the number of anchor
            boxes per anchor aka the number of potential boxes per anchor point.
            I was only passing the dimensions of all potential boxes per anchor point
            This is also including those boxes which fall outside of the image

"""
from keras import Model
from keras import models
from keras import optimizers
from keras import Sequential
from keras import layers
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
import keras.backend as K
import keras.applications
from keras import applications
from keras import utils
import cv2
import numpy as np
import os
import math

config = {
    "PathToImageAndGroundTruths" : "1. Data Gen//1. Data"
    ,"VGG16InputSize" : (224,224)
    ,"DefaultStride" : 16 # Default stride from original implementation of VGG16, standard length
    ,"AnchorBox" : {
        "AspectRatioW_div_W" : [1/3,1/2,3/4,1]
        ,"Scales" : [1/2,3/4,1,3/2]
    }
    ,"LabelSettings" : {
        "IOU_for_object": 0.7
        ,"IOU_for_background" : 0
    }
}

def pre_process_image_for_vgg(img):
    """
        Resizes the image to input of VGGInputSize specified in the config dictionary
        Normalises the image
        Reshapes the image to an array of images e.g. [[img],[img],..]

        If img has a shape of
    """
    if type(img) == np.ndarray: # Single image 
        resized_img = cv2.resize(img,config["VGG16InputSize"],interpolation = cv2.INTER_AREA)
        normalised_image = applications.vgg16.preprocess_input(resized_img)
        reshaped_to_array_of_images = np.array([normalised_image])
        return reshaped_to_array_of_images
    elif type(img) == list: # list of images
        img_list = img
        resized_img_list = [cv2.resize(image,config["VGG16InputSize"],interpolation = cv2.INTER_AREA) for image in img_list]
        resized_img_array = np.array(resized_img_list)
        normalised_images_array = applications.vgg16.preprocess_input(resized_img_array)
        return normalised_images_array



def get_data_to_cv2_np_array(img_path):
    """
        Reads data and returns it as cv2 numpy image
    """
    img = cv2.imread(image_path)
    return img

def generate_all_potential_bounding_boxes(settings,features_dimensions,input_dimensions,feature_to_input_scale):
    """
        Generates a set of anchor boxes (all potential bounding boxes that could be chosen)
        for a given point. It will be up to the RPN to decide if these boxes should be used
        
        Assumption 1: Settings will have the following attributes
            AspectRatioW_div_W: A list of float values representing the aspect ratios of
                the anchor boxes at each location on the feature map
            Scales: A list of float values representing the scale of the anchor boxes
                at each location on the feature map.
        Assumption 2: features_dimensions is a tuple of two positive integers
            features_dimensions = (CNN output feature map width, CNN output feature map height)
        Assumption 3: input_dimensions is a tuple of two positive integers
            input_dimensions = (input width, input height)
        Assumption 4: feature_to_input_scale is a tuple of two positive integers
            feature_to_input_scale = (scale from CNN output to input image width, scale from CNN output to input image height)
    """
    # Extract variables
    features_width, features_height = features_dimensions
    input_width, input_height = input_dimensions
    w_stride, h_stride = feature_to_input_scale

    # For the feature map (x,y) determine input image (x,y) as array 
    feature_to_input_coords_x  = [int(x_feature*w_stride) for x_feature in range(features_width)]
    feature_to_input_coords_y  = [int(y_feature*h_stride) for y_feature in range(features_height)]
    centre_coordinates_of_anchor_boxes = [[x,y] for x in feature_to_input_coords_x for y in feature_to_input_coords_y]
    
    pass 

def generate_potential_box_dimensions(settings,feature_to_input_x,feature_to_input_y):
    """
        Generate potential boxes height & width for each point aka anchor boxes given the 
        ratio between feature map to input scaling for x and y
        Assumption 1: Settings will have the following attributes
            AspectRatioW_div_W: A list of float values representing the aspect ratios of
                the anchor boxes at each location on the feature map
            Scales: A list of float values representing the scale of the anchor boxes
                at each location on the feature map.
    """
    box_width_height = []
    for scale in settings["Scales"]:
        for aspect_ratio_w_div_h in settings["AspectRatioW_div_W"]:
            width = round(feature_to_input_x*scale*aspect_ratio_w_div_h)
            height = round(feature_to_input_y*scale/aspect_ratio_w_div_h)
            box_width_height.append({"Width":width,"Height":height})
    return box_width_height

def generate_potential_boxes_for_coord(box_width_height,coord):
    """
        Assumption 1: box_width_height is an array of dictionary with each dictionary consisting of
            {"Width":positive integer, "Height": positive integer}
        Assumption 2: coord is an array of dictionary with each dictionary consistening of
            {"x":centre of box x coordinate,"y",centre of box y coordinate"}
    """
    potential_boxes = []
    for box_dim in box_width_height:
        potential_boxes.append({
            "x1": coord["x"]-int(box_dim["Width"]/2)
            ,"y1": coord["y"]-int(box_dim["Height"]/2)
            ,"x2": coord["x"]+int(box_dim["Width"]/2)
            ,"y2": coord["y"]+int(box_dim["Height"]/2)
        })
    return potential_boxes

def display_overlayed_feature_map_and_all_potential_boxes(img,coordinate_of_anchor_boxes,potential_boxes,ground_truth=None,wait_time_ms=0):
    """
        Displays the centre points of where the undersized feature map would be on the image.
        Resized image to input size of neural net
        Overlayed potential boxes aka anchors
        Overlays centre of where the feature map aka output size of CNN would be
        resizes to something appropriate for screen
        If grouth_truth is given then the ground truth box is 
        Displays image until user clicks on image and presses any key to continue the program

        Assumption 1: Only one ground truth box
    """
    display_img = img.copy()

    if ground_truth != None:
        start_point = (ground_truth["x1"],ground_truth["y1"])
        end_point = (ground_truth["x2"],ground_truth["y2"])
        cv2.rectangle(display_img,start_point,end_point,(255,255,255),3)

    display_img = cv2.resize(display_img,config["VGG16InputSize"],interpolation = cv2.INTER_AREA)

    for box in potential_boxes:
        start_point = (box["x1"],box["y1"])
        end_point = (box["x2"],box["y2"])
        cv2.rectangle(display_img,start_point,end_point,(255,0,0),1)

    for coord in coordinate_of_anchor_boxes:
        cv2.circle(display_img,(coord["x"],coord["y"]),1,(0,0,255))


    cv2.imshow("Potential Boxes",display_img)
    cv2.waitKey(wait_time_ms)
    

def read_img_labels(img_label_path):
    """
        Reads the csv file which has bounding boxes as the labels
        Assumption 1: Only 1 object per CSV
    """
    with open(img_label_path,"r") as img_label_file:
        data_lines = img_label_file.readlines()
        x1,y1,x2,y2 = data_lines[1].split("\t")
        img_label_file.close()
    return {"x1":int(x1),"y1":int(y1),"x2":int(x2),"y2":int(y2)}

def get_iou(bb1, bb2):
    """
        Gets the Intersection Over Area, aka how much they cross over
        Assumption 1: Each box is a dictionary with the following
            {"x1":top left top corner x coord,"y1": top left top corner y coord,"x2": bottom right corner x coord,"y2":bottomr right corner y coord}
    """
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

def is_box_in_image_bounds(input_image_shape,box):
    """
        Determines if the box is in the image and returns True or False
        Assumption 1: input_image_shape is a tuple of (img height,img width,channels)
        Assumption 2: box is a dictionary having keys of ["x1","y1","x2","y2"]
        Assumption 3: box 1 coordinate is the top left and 2 is the bottom right
    """
    assert box['x1'] < box['x2']
    assert box['y1'] < box['y2']
    width, height, _ = input_image_shape
    if box["x1"] < 0:
        return False
    if box["y1"] < 0:
        return False
    if box["x2"] >= width:
        return False
    if box["y2"] >= height:
        return False
    return True

def loss_class(y_true,y_predict):
    """
        A custom loss function used to compile the region proposal network that determines if label of background or
        foreground is correct

        y_true: The true value
        y_predict: The predicted value
    """
    condition = K.not_equal(y_true, -1)
    indices = K.tf.where(condition)

    target = K.tf.gather_nd(y_true, indices)
    output = K.tf.gather_nd(y_predict, indices)

    loss = K.binary_crossentropy(target, output)
    return K.mean(loss)

def smoothL1(y_true, y_predict):
    """
    TODO decide whether or not you can go straight to k.tf.losses
    """
    nd=K.tf.where(K.tf.not_equal(y_true,0))
    y_true=K.tf.gather_nd(y_true,nd)
    y_predict=K.tf.gather_nd(y_predict,nd)
    x = K.tf.losses.huber_loss(y_true,y_predict)
    return x

def create_region_proposal_network(number_of_potential_bounding_boxes,number_of_feature_map_channels=512):
    """
        Creates the region proposal network which takes the input of the feature map and 
        Compiles the model and returns it

        RPN consists of an input later, a CNN and two output layers.
            output_deltas: 
            output_scores:

        Note: Number of feature map channels should be the last element of model.predict().shape
    """
    # Input layer
    feature_map_tile = layers.Input(shape=(None,None,number_of_feature_map_channels),name="RPN_Input_Same")
    # CNN component, ensure that padding is the same so that it has the same dimensions as input feature map
    convolution_3x3 = layers.Conv2D(filters=512,kernel_size=(3, 3),name="3x3",padding="same")(feature_map_tile)
    # Output layers
    output_deltas = layers.Conv2D(filters= 4 * number_of_potential_bounding_boxes,kernel_size=(1, 1),activation="linear",kernel_initializer="uniform",name="Output_Deltas")(convolution_3x3)
    output_scores = layers.Conv2D(filters=1 * number_of_potential_bounding_boxes,kernel_size=(1, 1),activation="sigmoid",kernel_initializer="uniform",name="Output_Prob_FG")(convolution_3x3)

    model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])

    # TODO add loss_cls and smoothL1
    model.compile(optimizer='adam', loss={'scores1':losses.binary_crossentropy, 'deltas1':losses.huber})

    return model

def get_all_files_in_directory_with_ext(directory_to_search:str, valid_file_extensions:list):
    """
        uses os.walk to go through the file and returns a list with the full file paths that have the valid file extensions
        Assumption 1: valid_file_extensions is a list of all LOWER CASE file extensions with the dot
            e.g. [".csv",".jpg"]
    """
    full_file_paths_list = []
    for root, dirs, files in os.walk(directory_to_search, topdown=False):
        for f in files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in valid_file_extensions:
                full_file_paths_list.append(os.path.join(root, f))
    return full_file_paths_list

def get_image_files_and_corresponding_csv_files(directory_to_search:str):
    """
        Get a list of all image files, then csv files and compare if they have the same base file name.
        If they do, join them in a list of dictionaries
    """
    all_image_file_paths = get_all_files_in_directory_with_ext(directory_to_search, [".jpg",".png"])
    all_csv_file_paths = get_all_files_in_directory_with_ext(directory_to_search, [".csv"])
    data_files_list = []
    for img_path in all_image_file_paths:
        img_file_base_name = os.path.splitext(img_path)[0]
        csv_files_with_same_base_name = [f for f in all_csv_file_paths if img_file_base_name == os.path.splitext(f)[0]]
        if len(csv_files_with_same_base_name) > 0:
            data_files_list.append({
                "ImgPath":img_path
                ,"HasCSV" : True
                ,"CSVPath" : csv_files_with_same_base_name[0]
            })
        else:
            data_files_list.append({
                "ImgPath":img_path
                ,"HasCSV" : False
                ,"CSVPath" : None
            })
    return data_files_list

def find_feature_map_to_input_scale_and_offset(pre_processed_input_image,feature_maps):
    """
        Finds the scale and offset from the feature map (output) of the CNN classifier to the pre-processed input image of the CNN
    """
    # Find shapes of feature maps and input images to the classifier CNN
    input_image_shape = pre_processed_input_image.shape
    feature_map_shape = feature_maps.shape
    img_height, img_width, _ = input_image_shape
    features_height, features_width, _ = feature_map_shape

    # Find mapping from features map (output of vggmodel.predict) back to the input image
    feature_to_input_x = img_width / features_width
    feature_to_input_y = img_height / features_height

    # Put anchor points in the centre of 
    feature_to_input_x_offset = feature_to_input_x/2
    feature_to_input_y_offset = feature_to_input_y/2

    return feature_to_input_x, feature_to_input_y, feature_to_input_x_offset, feature_to_input_y_offset

def get_get_coordinates_of_anchor_points(feature_map,feature_to_input_x,feature_to_input_y,x_offset,y_offset):
    """
        Maps the CNN output (Feature map) coordinates on the input image to the CNN 
        Returns the coordinates as a list of dictionaries with the format {"x":x,"y":y}
    """
    features_height, features_width, _ = feature_map.shape

    # For the feature map (x,y) determine the anchors on the input image (x,y) as array 
    feature_to_input_coords_x  = [int(x_feature*feature_to_input_x+x_offset) for x_feature in range(features_width)]
    feature_to_input_coords_y  = [int(y_feature*feature_to_input_y+y_offset) for y_feature in range(features_height)]
    coordinate_of_anchor_points = [{"x":x,"y":y} for x in feature_to_input_coords_x for y in feature_to_input_coords_y]

    return coordinate_of_anchor_points

def get_potential_boxes_for_region_proposal(pre_processed_input_image,feature_maps,feature_to_input_x, feature_to_input_y, x_offset, y_offset):
    """
        Generates the anchor points (the centre of the enlarged feature map) as an (x,y) position on the input image
        Generates all the potential bounding boxes for each anchor point
        Removes all potential bounding boxes that are outside the image
        returns a list of potential bounding boxes in the form {"x1","y1","x2","y2"}
    """
    # Find shapes of input images to the classifier CNN
    input_image_shape = pre_processed_input_image.shape

    # For the feature map (x,y) determine the anchors on the input image (x,y) as array 
    coordinate_of_anchor_boxes = get_get_coordinates_of_anchor_points(feature_maps,feature_to_input_x,feature_to_input_y,x_offset,y_offset)

    # Create potential boxes for classification
    boxes_width_height = generate_potential_box_dimensions(config["AnchorBox"],feature_to_input_x,feature_to_input_y)
    list_of_potential_boxes_for_coords = [generate_potential_boxes_for_coord(boxes_width_height,coord) for coord in coordinate_of_anchor_boxes]
    potential_boxes = [box for boxes_for_coord in list_of_potential_boxes_for_coords for box in boxes_for_coord]
    potential_boxes_in_img = [box for box in potential_boxes if is_box_in_image_bounds(input_image_shape,box)]
    
    return potential_boxes_in_img, potential_boxes

def get_scaled_ground_truth_bounding_box(original_img,pre_processed_input_img):
    """
        Get the scaled ground truth the original image to being resized to the CNN classifier required input
    """
    original_height, original_width, _ = original_img.shape
    pre_processed_img_height, pre_processed_img_width, _ = pre_processed_input_img.shape
    x_scale = original_width / pre_processed_img_width
    y_scale = original_height / pre_processed_img_height
    scaled_ground_truth_box = {
        "x1" : round(ground_truth_box["x1"]/x_scale)
        ,"y1" : round(ground_truth_box["y1"]/y_scale)
        ,"x2" : round(ground_truth_box["x2"]/x_scale)
        ,"y2" : round(ground_truth_box["y2"]/y_scale)
    }
    return scaled_ground_truth_box

def get_foreground_and_background_labels(scaled_ground_truth_box,potential_boxes_in_img,label_set_size=256,background_iou_thresh=0.0,foreground_iou_thresh=0.7):
    """
        Gets a set of labelled foreground and background boxes
        First, loops through all the potential boxes in the image and checks if the IOU is greater or equal to the threshold
            it labels these as foreground
        Second, checks to see if there were any potential boxes that had an IOU greater or equal to the threshold. 
            If none were detected, it finds the next best option with max IOU
            If there are still no foreground labels an error is raised
        Third, adds background labels which have an IOU of less than or equal to background_iou_thresh up to the label_set_size
            Assumption 1: background_iou_thresh will be a float between 0 - 1
                          background regions will be those will an IOU less than background_iou_thresh
            Assumption 2: foreground_iou_thresh will be a float between 0 - 1
                          foreground regions will be those will an IOU more than foreground_iou_thresh
            Assumption 3: If there are no proposed regions with an IOU with the ground truth above foreground_iou_thresh
                          then the next best option is taken as long as the IOU is above 0. 
            Assumption 4: There is only one object per image, aka only 1 ground truth box per image
    """
    # Make computation faster by using list comprehension
    iou_box_with_gtruth = [get_iou(scaled_ground_truth_box,box) for box in potential_boxes_in_img]

    # Generate foreground aka object labels from thresholds, stop after label_set_size/2
    foreground_box_labels = []
    for index, potential_box in enumerate(potential_boxes_in_img):
        if iou_box_with_gtruth[index] >= foreground_iou_thresh:
            foreground_box_labels.append(potential_box)
        if len(foreground_box_labels) > label_set_size/2:
            break
    
    # If no potential above IOU threshold then pick the next best thing
    # This was likely to happen in my dataset
    if len(foreground_box_labels) == 0:
        index_of_box_with_max_iou = [index for index, iou in enumerate(iou_box_with_gtruth) if iou == max(iou_box_with_gtruth)][0]
        assert index_of_box_with_max_iou != 0 # Raise error if this happens
        best_potential_box = potential_boxes_in_img[index_of_box_with_max_iou]
        foreground_box_labels.append(best_potential_box)

    # Generate background aka not object labels from thresholds
    background_box_labels = []
    for index, potential_box in enumerate(potential_boxes_in_img):
        if iou_box_with_gtruth[index] <= background_iou_thresh:
            background_box_labels.append(potential_box)
        if len(background_box_labels) + len(foreground_box_labels) >= label_set_size:
            break

    return foreground_box_labels, background_box_labels

def find_corresponding_anchor_point_from_box(coordinate_of_anchor_points,box):
    """
        Assumption 1: The centre of the box will correspond to the 
    """
    pass

def convert_x1y1x2y2_to_XcYcWH(box):
    """
        Convert box from dictionary of {"x1":,"y1":,"x2":,"y2"} to {"x_centre":,"y_centre":,"width":,"height":}
        Assumption 1: point 1 is the top left and point 2 is the bottom right hand corner
    """
    assert box["x1"] <= box["x2"]
    assert box["y1"] <= box["y2"]
    width = box["x2"] - box["x1"]
    height = box["y2"] - box["y1"]
    x_centre = round(box["x1"] + width/2)
    y_centre = round(box["y1"] + height/2)
    return {"x_centre":x_centre,"y_centre":y_centre,"width":width,"height":height}

def convert_bounding_box_dict_to_numpy_array(box):
    """
        Utility function to convert the bounding box from a dictionary of  {"x1":,"y1":,"x2":,"y2"} or 
        {"x_centre":,"y_centre":,"width":,"height":} to a numpy array of ["x_centre","y_centre","width","height"]
        Assumption 1: point 1 is the top left and point 2 is the bottom right hand corner
    """
    keys = box.keys()
    if list(keys) == ["x1","y1","x2","y2"]:
        box = convert_x1y1x2y2_to_XcYcWH(box)

    return np.array([box["x_centre"],box["y_centre"],box["width"],box["height"]])

def get_feature_map_coordinates_from_potential_box(box,input_to_feature_x_scale,input_to_feature_y_scale,input_to_feature_x_offset,input_to_feature_y_offset):
    """
        Convert box of either {"x1":,"y1":,"x2":,"y2"} or {"x_centre":,"y_centre":,"width":,"height":} to 
        the feature map coordinates
    """
    keys = box.keys()
    if list(keys) == ["x1","y1","x2","y2"]:
        box = convert_x1y1x2y2_to_XcYcWH(box)
    
    feature_x_coord = round( (box["x_centre"] + input_to_feature_x_offset) * input_to_feature_x_scale )
    feature_y_coord = round( (box["y_centre"] + input_to_feature_y_offset) * input_to_feature_y_scale )

    return feature_x_coord,feature_y_coord
    
def generate_data_set_for_training(all_img_and_csv_files_list,img_and_csv_files_list,potential_boxes_in_img,background_iou_thresh=0.0,foreground_iou_thresh=0.7):
    """
        Assumption 1: Each proposed box is unique and hence through x1,y1,x2,y2 can be mapped back to the anchor point
                      aka CNN output feature map "pixel" 
    """

    for index, img_and_csv_file in enumerate(img_and_csv_files_list): 
        img = cv2.imread(img_and_csv_file["ImgPath"])
        ground_truth_box = read_img_labels(img_and_csv_file["CSVPath"])
        scaled_ground_truth_box = get_scaled_ground_truth_bounding_box(img,array_of_prediction_ready_images[index])
        foreground_box_labels_for_img, background_box_labels_for_img = get_foreground_and_background_labels(scaled_ground_truth_box,potential_boxes_in_img)

    pass

def convert_rpn_output_to_bounding_box(rpn_output):
    """
        Takes the output of the Regional Proposal Network (RPN) and convert it into a bounding box. 
        TODO fix this later
    """
    predicted_scores_for_anchor_boxes, predicted_adjustments = rpn_output
    xy_centre_wh_predicted_adjustments = predicted_adjustments.reshape(-1,4)
    flattened_predicted_scores_for_anchor_boxes = predicted_scores_for_anchor_boxes.reshape(-1,1)

    threshold = 0.95
    number_of_potential_boxes = len(potential_boxes)
    for pred_index, prediction in enumerate(flattened_predicted_scores_for_anchor_boxes):
        if prediction >= threshold and pred_index > number_of_potential_boxes and pred_index < 1400:
            # Indexes to select scores and delta movements
            selected_bounding_box = pred_index % number_of_potential_boxes
            potential_anchor_box = potential_boxes[selected_bounding_box]

            # Selected Anchor Boxes
            anchor_box_width = potential_anchor_box["x2"] - potential_anchor_box["x1"]
            anchor_box_x_centre = potential_anchor_box["x1"] + anchor_box_width/2
            anchor_box_height = potential_anchor_box["y2"] - potential_anchor_box["y1"]
            anchor_box_y_centre = potential_anchor_box["y1"] + anchor_box_height/2

            # Suggested Delta Movements
            delta_x_centre = float(xy_centre_wh_predicted_adjustments[pred_index][0])
            delta_y_centre = float(xy_centre_wh_predicted_adjustments[pred_index][1])
            delta_width = xy_centre_wh_predicted_adjustments[pred_index][2]
            delta_height = xy_centre_wh_predicted_adjustments[pred_index][3]

            # Generate predicted location
            # Centre formula = anchor_box_centre + change_in_centre_aka_delta x anchor_box_wdith
            # Length of side = anchor_side + e^(change_in_side_aka_delta) x anchor_side # Don't forget to use the exponential
            predicted_x_centre = int(round(anchor_box_x_centre + delta_x_centre * anchor_box_width,0))
            predicted_y_centre = int(round(anchor_box_y_centre + delta_y_centre * anchor_box_height,0))
            predicted_width = int(round( anchor_box_width + math.exp(delta_width) * anchor_box_width ,0))
            predicted_height = int(round( anchor_box_height + math.exp(delta_height) * anchor_box_height ,0))
    pass

if __name__ == "__main__":
    print("starting...")
    # Get all image and ground truth labels file paths 
    all_img_and_csv_files_list = get_image_files_and_corresponding_csv_files(config["PathToImageAndGroundTruths"])
    img_and_csv_files_list = [v for v in all_img_and_csv_files_list if v["HasCSV"] == True]

    # Get vgg model
    vggmodel = applications.VGG16(include_top=False,weights='imagenet') 

    # Extract features for images (used dictionary comprehension to stop getting warning messages from Keras)
    list_of_images = [cv2.imread(img_and_csv_file["ImgPath"]) for img_and_csv_file in img_and_csv_files_list]
    array_of_prediction_ready_images = pre_process_image_for_vgg(list_of_images)
    array_of_feature_maps = vggmodel.predict(array_of_prediction_ready_images)

    # Find conversions from feature map (CNN output) to input image
    feature_to_input_x_scale, feature_to_input_y_scale, feature_to_input_x_offset, feature_to_input_y_offset = find_feature_map_to_input_scale_and_offset(array_of_prediction_ready_images[0],array_of_feature_maps[0])

    # Find conversions from input image to feature map (CNN output)
    input_to_feature_x_scale = 1/feature_to_input_x_scale
    input_to_feature_y_scale = 1/feature_to_input_y_scale
    input_to_feature_x_offset = -feature_to_input_x_offset
    input_to_feature_y_offset = -feature_to_input_y_offset

    coordinate_of_anchor_points = get_get_coordinates_of_anchor_points(array_of_feature_maps[0],feature_to_input_x_scale,feature_to_input_y_scale,feature_to_input_x_offset, feature_to_input_y_offset)
    potential_boxes_in_img, potential_boxes = get_potential_boxes_for_region_proposal(array_of_prediction_ready_images[0],array_of_feature_maps[0],feature_to_input_x_scale, feature_to_input_y_scale, feature_to_input_x_offset, feature_to_input_y_offset)

    # Create region proposal network
    # TODO update create_reion_proposal network
    rpn_model = create_region_proposal_network(len(potential_boxes))

    max_iou_calculated = []
    human_readable_dataset = []
    for index, img_and_csv_file in enumerate(img_and_csv_files_list): 
        img = cv2.imread(img_and_csv_file["ImgPath"])
        ground_truth_box = read_img_labels(img_and_csv_file["CSVPath"])
        scaled_ground_truth_box = get_scaled_ground_truth_bounding_box(img,array_of_prediction_ready_images[index])

        foreground_labels, background_labels = get_foreground_and_background_labels(scaled_ground_truth_box,potential_boxes_in_img)

        for foreground_label in foreground_labels: 
            feature_x_coord,feature_y_coord = get_feature_map_coordinates_from_potential_box(foreground_label,input_to_feature_x_scale,input_to_feature_y_scale,input_to_feature_x_offset,input_to_feature_y_offset)

            human_readable_dataset.append({
                "FeatureMap": array_of_feature_maps[index]
                ,"Label":"foreground"
                ,"ProposedBox" : foreground_label
                ,"GroundTruth" :  scaled_ground_truth_box
                ,"ImagePath" :img_and_csv_file["ImgPath"]
                })

        for background_label in background_labels: 
            feature_x_coord,feature_y_coord = get_feature_map_coordinates_from_potential_box(background_label,input_to_feature_x_scale,input_to_feature_y_scale,input_to_feature_x_offset,input_to_feature_y_offset)

            # I don't really understand what is going on here
            padded_feature_map = np.pad([array_of_feature_maps[index]],((0,0),(1,1),(1,1),(0,0)),mode='constant')
            padded_feature_map = np.squeeze(padded_feature_map)
            feature_map_tile = padded_feature_map[feature_y_coord:feature_y_coord+3,feature_x_coord:feature_x_coord+3]
            
            human_readable_dataset.append({
                "FeatureMap": array_of_feature_maps[index]
                ,"Label":"background"
                ,"ProposedBox" : background_label
                ,"GroundTruth" :  scaled_ground_truth_box
                ,"ImagePath" :img_and_csv_file["ImgPath"]
                })
        #display_overlayed_feature_map_and_all_potential_boxes(img,coordinate_of_anchor_points,foreground_labels,ground_truth=ground_truth_box)

    # Get data as 50 50 split, assumption is that there is not many foregrounds labels
    number_of_foreground_labels = len([data for data in human_readable_dataset if data["Label"]=="foreground"])

    # build dataset for neural network
    proposed_box = []

    for data in human_readable_dataset:
        proposed_box.append(convert_bounding_box_dict_to_numpy_array(data["ProposedBox"]))
        
    prediction_feature_map = np.array([array_of_feature_maps[0]])
    original_img = list_of_images[0]
    # cv2.imshow("original_img",original_img)
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

    # Output following (height, width, anchor_num)     (height, width, anchor_num * 4)
    rpn_output  = rpn_model.predict(prediction_feature_map)    



            # Debugging
            print(f"Val[{pred_index}]={prediction}, AnchorBox[{selected_bounding_box}] = [{anchor_box_x_centre},{anchor_box_y_centre},{anchor_box_width},{anchor_box_height}]")
            print(f"Val[{pred_index}]={prediction}, DeltaMovements=[{delta_x_centre},{delta_y_centre},{delta_width},{delta_height}]")
            print(f"Val[{pred_index}]={prediction}, Final=[{predicted_x_centre},{predicted_y_centre},{predicted_width},{predicted_height}]")
    

    
    print("finishing...")