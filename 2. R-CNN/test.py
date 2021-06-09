from keras import Model
from keras import models
from keras import optimizers
from keras import Sequential
from keras import layers
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K
import keras.applications
from keras import applications
from keras import utils
import cv2
import numpy as np
import os
import math

config = {
    "ImgPath" : "1. Data Gen\\1. Data\\1X9A1712.jpg" #"Put your image path here"
    ,"VGG16InputSize" : (224,224)
    ,"AnchorBox" : {
        "AspectRatioW_div_W" : [1/3,1/2,3/4,1]
        ,"Scales" : [1/2,3/4,1,3/2]
    }
}

def main(): ############ MAIN FUNCTION - START HERE ############
    # Get vgg model
    vggmodel = applications.VGG16(include_top=False,weights='imagenet') 

    # Extract features for images (used dictionary comprehension to stop getting warning messages from Keras)
    list_of_images = [cv2.imread(config["ImgPath"])]
    array_of_prediction_ready_images = pre_process_image_for_vgg(list_of_images)
    array_of_feature_maps = vggmodel.predict(array_of_prediction_ready_images)

    # Find conversions from feature map (CNN output) to input image
    feature_to_input_x_scale, feature_to_input_y_scale, feature_to_input_x_offset, feature_to_input_y_offset = find_feature_map_to_input_scale_and_offset(array_of_prediction_ready_images[0],array_of_feature_maps[0])

    # get potential boxes, aka anchor boxes
    potential_boxes = get_potential_boxes_for_region_proposal(array_of_prediction_ready_images[0],array_of_feature_maps[0],feature_to_input_x_scale, feature_to_input_y_scale, feature_to_input_x_offset, feature_to_input_y_offset)

    # Create region proposal network
    rpn_model = create_region_proposal_network(len(potential_boxes))

    # Output following (height, width, anchor_num)     (height, width, anchor_num * 4)
    predicted_scores_for_anchor_boxes, predicted_adjustments  = rpn_model.predict(array_of_feature_maps)

    print(f"predicted_scores_for_anchor_boxes.shape = {predicted_scores_for_anchor_boxes.shape}, predicted_adjustments.shape = {predicted_adjustments.shape}")
    print(f"But why is there the ,5,5, bit? I don't know which ones to choose now to get the predicted bounding box?")
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
    
    return potential_boxes

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
    # CNN component
    convolution_3x3 = layers.Conv2D(filters=512,kernel_size=(3, 3),name="3x3")(feature_map_tile)
    # Output layers
    output_deltas = layers.Conv2D(filters= 4 * number_of_potential_bounding_boxes,kernel_size=(1, 1),activation="linear",kernel_initializer="uniform",name="Output_Deltas")(convolution_3x3)
    output_scores = layers.Conv2D(filters=1 * number_of_potential_bounding_boxes,kernel_size=(1, 1),activation="sigmoid",kernel_initializer="uniform",name="Output_Prob_FG")(convolution_3x3)

    model = Model(inputs=[feature_map_tile], outputs=[output_scores, output_deltas])

    # TODO add loss_cls and smoothL1
    model.compile(optimizer='adam', loss={'scores1':losses.binary_crossentropy, 'deltas1':losses.huber})

    return model

if __name__ == "__main__":
    main()