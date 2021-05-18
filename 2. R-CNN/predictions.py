from keras import Model, models, applications
import cv2
import tensorflow as tf # Need this to run
import numpy as np
import os
import pickle
"""
    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Purpose:
        Load the already trained model and test against another image

    Lessons Learnt
        1) Must have tensorflow imported otherwise it will not run because of the lambda layer which comes from the tensorflow module
        2) When creating a pitch black zeros cv2 image use np.zeroes((width,height),dtype=np.uint8), it defaults to float which is between 0 and 1 not 0 - 255 like what you're used to
        3) How to normalise an array of uint8
            https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values/46689933
        4) How to multiple a non-binary array to opencv
            https://stackoverflow.com/questions/49879474/how-to-apply-a-non-binary-mask-on-an-image-in-opencv
        5) How to deal with multiple boxes of similar area. 
            Use non maximal surrpession 
            https://www.pyimagesearch.com/2014/11/17/non-maximum-suppression-object-detection-python/
"""

config = {
    "ModelPath" : "2. Models\\weights-improvement-20-0.90.hdf5"
    ,"ClassifierInput" : (224,224)
    ,"ModelInput" : (224,224)
    ,"OutputSaveLocation" : "1. Data Gen\\3. Data From Predictions"
    ,"InputImageFiles" : "1. Data Gen\\2. Data for Predictions"
    ,"ValidImageFileExtensions" : [".jpg",".png"]
    ,"ValidPredictionFileExtensions" : [".p"]
    ,"ImageThreshold" : 0.98
    ,"IouThreshold" : 0.5
}

def get_resized_img_to_model(image,box,resize_dimensions=(224,224)):
    x,y,w,h = box
    cropped_resized_img = cv2.resize(img[y:y+h,x:x+w], resize_dimensions, interpolation = cv2.INTER_AREA)
    return cropped_resized_img

def make_prediction(cropped_resized_img,model):
    preprocessed_img = applications.vgg16.preprocess_input(cropped_resized_img).reshape(1, 224, 224, -1) # Must have how many images to process
    prediction = model.predict(preprocessed_img)
    return prediction

def change_box(base_image,box,change_array):
    """
        Assumption 1: Contents of box are as follows
            [x1 ,y2 ,width ,height]
    """
    height, width, _ = base_image.shape

    new_box = [0,0,0,0]
    for i,value in enumerate(change_array):
        if value != 0:
            new_box[i] = box[i] + value
        else:
            new_box[i] = box[i]

    assert new_box[0] >= 0
    assert new_box[1] >= 0
    assert new_box[0]+new_box[2] <= width
    assert new_box[1]+new_box[3] <= height

    return new_box

def potential_steps(base_image, original_box,pixel_step_size = 5):
    possible_directions = []
    for step_size in range(pixel_step_size):
        for i in  range(4):
            temp = [0,0,0,0]
            temp[i] = 1*step_size
            possible_directions.append(temp)
        for i in  range(4):
            temp = [0,0,0,0]
            temp[i] = -1*step_size
            possible_directions.append(temp)

    potential_boxes = []
    for step in possible_directions:
        try:
            potential_boxes.append(change_box(base_image,original_box,step))
        except:
            pass

    return potential_boxes

def get_nearest_box(img, previous_predictions_list,previous_boxes):
    back_max, fore_max = previous_predictions_list.max(axis=0)
    max_index = [i for i,v in enumerate(previous_predictions_list) if v[1] == fore_max][0]
    chosen_box_to_search = previous_boxes[max_index]

    potential_boxes = potential_steps(img,chosen_box_to_search,pixel_step_size = 1)
    potential_cropped_images = [get_resized_img_to_model(img,box,resize_dimensions=config["ClassifierInput"]) for box in potential_boxes]
    potential_steps_prediction_ready_images = np.array([applications.vgg16.preprocess_input(img) for img in potential_cropped_images])
    potential_steps_predictions = model.predict(potential_steps_prediction_ready_images)

    previous_max_fore = fore_max
    back_max, fore_max = potential_steps_predictions.max(axis=0)
    max_index = [i for i,v in enumerate(potential_steps_predictions) if v[1] == fore_max][0]

    chosen_box_to_search = potential_boxes[max_index]
    return chosen_box_to_search, fore_max, potential_steps_predictions, potential_boxes

def create_mask(base_img,box,pixel_colour:int):
    height, width, _ = base_img.shape
    x,y,w,h = box
    mask = np.zeros((height, width),dtype=np.uint8)
    mask[y:y+h, x:x+w] = pixel_colour
    return mask

def get_list_of_images_not_predicted():
    """
        Get those images in the to be predicted folder that are images and make predictions based upon those that do not already have predictions

        Assumptions
            1) Each image name is unique
            2) the image predictions is in another folder with the same basename as the image file but with a different file extension, this time .p
    """
    list_of_imgs = []
    for root, dirs, files in os.walk(config["InputImageFiles"], topdown=False):
        for f in files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in config["ValidImageFileExtensions"]:
                list_of_imgs.append(os.path.join(root, f))

    list_of_img_predictions = []
    for root, dirs, files in os.walk(config["OutputSaveLocation"], topdown=False):
        for f in files:
            ext = os.path.splitext(f)[-1].lower()
            if ext in config["ValidPredictionFileExtensions"]:
                list_of_img_predictions.append(os.path.join(root, f))

    list_of_imgs_with_no_saved_predictions = []
    for img_full_file_name in list_of_imgs:
        img_ext = os.path.splitext(img_full_file_name)[1].lower()
        img_base_name = os.path.basename(img_full_file_name)[:-len(img_ext)]

        prediction_base_file_names = [os.path.basename(v)[:-len(".p")] for v in list_of_img_predictions]
        if not img_base_name in prediction_base_file_names:
            list_of_imgs_with_no_saved_predictions.append(img_full_file_name)

    return list_of_imgs_with_no_saved_predictions 

def display_selected_regions(img,predictions,rects,threshold_probability):
    """
        Description: Orders the bounding boxes to the probability of finding a foreground and
                    displays them one by one for the user to see. The user must press any key to progress.
        Assumption 1: There are objects in the image that mean bounding boxes are selected which meet a 
                    threshhold greater than the one specified
    """
    _, foreground_max = predictions.max(axis=0)
    max_index = [i for i,v in enumerate(predictions) if v[1] == foreground_max][0]

    foreground_predictions_with_index = [[i,v[1]] for i,v in enumerate(list(predictions))]
    sorted_foreground_predictions_with_index = sorted(foreground_predictions_with_index, reverse=True,key=lambda x: x[1])

    number_of_predictions_above_threshold = len([v for v in list(predictions) if v[1] > threshold_probability])

    draw_img = img.copy()
    for index, prediction in sorted_foreground_predictions_with_index[:number_of_predictions_above_threshold]:
        x,y,w,h = rects[index]
        resized_cropped_image = cv2.resize(draw_img[y:(y+h),x:(x+w)], (244,244), interpolation = cv2.INTER_AREA)
        cv2.rectangle(draw_img, (x,y), (x+w,y+h), (255,255,255), 4) 
        cv2.putText(draw_img,f"Prob of My Face = {prediction}", 
            (100,100), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255,255,255),
            2)
        
        cv2.imshow("BaseImage",draw_img)
        cv2.imshow("ResizedImage",resized_cropped_image)
        cv2.waitKey(0)
        draw_img = img.copy()
    cv2.destroyAllWindows()

def display_heat_map_of_object(img,predictions,rects,threshold_probability):
    """
        Description: Creates a heatmap of the highest probabilities (all asume to be equal importance)
    """
    _, foreground_max = predictions.max(axis=0)
    max_index = [i for i,v in enumerate(predictions) if v[1] == foreground_max][0]

    foreground_predictions_with_index = [[i,v[1]] for i,v in enumerate(list(predictions))]
    sorted_foreground_predictions_with_index = sorted(foreground_predictions_with_index, reverse=True,key=lambda x: x[1])

    number_of_predictions_above_threshold = len([v for v in list(predictions) if v[1] > threshold_probability])

    pixel_colour = int(255/number_of_predictions_above_threshold)
    bounding_box_masks = [create_mask(img,rects[i],pixel_colour) for i,v in sorted_foreground_predictions_with_index[:number_of_predictions_above_threshold]]

    heat_map = sum(bounding_box_masks)

    cv2.imshow(f"Heat Map of Top {number_of_predictions_above_threshold} Search Areas",heat_map)
    cv2.imshow(f"Original image",img)
    cv2.waitKey(20000)
    cv2.destroyAllWindows()

def get_iou(bb1, bb2):
    # Convert for readablity
    bb1_x1, bb1_y1 = bb1[0], bb1[1]
    bb1_x2 = bb1_x1 + bb1[2]
    bb1_y2 = bb1_y1 + bb1[3]

    bb2_x1, bb2_y1 = bb2[0], bb2[1]
    bb2_x2 = bb2_x1 + bb2[2]
    bb2_y2 = bb2_y1 + bb2[3]

    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bb1_area = (bb1_x2 - bb1_x1) * (bb1_y2 - bb1_y1)
    bb2_area = (bb2_x2 - bb2_x1) * (bb2_y2 - bb2_y1)
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    
    return iou

def area_of_rect(bounding_box):
    """
        Finds the area of a bounding box produced from selective search
    """
    _, _, width, height = bounding_box # Convert for readablity
    return width * height

def find_area_of_intersection(bb1, bb2):
    """
        Finds the area of which the bounding boxes intersect
    """
    # Convert for readablity
    bb1_x1, bb1_y1 = bb1[0], bb1[1]
    bb1_x2 = bb1_x1 + bb1[2]
    bb1_y2 = bb1_y1 + bb1[3]

    bb2_x1, bb2_y1 = bb2[0], bb2[1]
    bb2_x2 = bb2_x1 + bb2[2]
    bb2_y2 = bb2_y1 + bb2[3]

    x_left = max(bb1_x1, bb2_x1)
    y_top = max(bb1_y1, bb2_y1)
    x_right = min(bb1_x2, bb2_x2)
    y_bottom = min(bb1_y2, bb2_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    return (x_right - x_left) * (y_bottom - y_top)

def find_top_n_images_to_cover_area(rects,number_of_rects=20):
    """
        Finds n rectangles that do not intersect and cover the most area of an image
    """ 
    rects_from_highest_to_lowest_area = sorted(rects,key=area_of_rect,reverse=True)

def divide_image_into_bounding_boxes(img,width_split=1,height_split=1):
    """
        This splits the image into width_split x height_split bounding boxes based upon the...
            width_split: how many times to split the image in the x direction
            height_split: how many times to split the image in the y direction
        Assumption 1: The width_split and height_split must be greater than 0 and integers
    """
    assert width_split > 0 and height_split > 0
    height, width, _ = img.shape

    height_div = int(height/height_split)
    width_div = int(width/width_split)

    bounding_boxes = []
    for x_i in range(width_split):
        for y_i in range(height_split):
            x = width_div * x_i
            y = height_div * y_i
            bounding_boxes.append([x,y,width_div,height_div])
    return bounding_boxes

def get_predictions_from_bounding_boxes(img,model,bounding_boxes):
    """
        One line wrapper to get prediction array from bounding boxes
        Assumption 1: bounding box is an array in the following form [x,y,w,h]
    """

    cropped_resized_images = [get_resized_img_to_model(img,box,resize_dimensions=config["ClassifierInput"]) for box in bounding_boxes]
    prediction_ready_arrays = np.array([applications.vgg16.preprocess_input(resized_img) for resized_img in cropped_resized_images])

    predictions = model.predict(prediction_ready_arrays)

    return predictions, cropped_resized_images, bounding_boxes

def non_max_suppression_fast(bounding_boxes, iou_threshhold):
    """
        When multiple similar bounding boxes are returned the final box can be calculated using non-maximal suppression
        This was taken from Py-imagesearch - https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

        
    """
    # if there are no boxes, return an empty list
    if len(bounding_boxes) == 0:
        return []

    # Convert into [x1,y1,x2,y2] 
    boxes = np.array([[box[0],box[1],box[0]+box[2],box[1]+box[3]] for box in bounding_boxes])
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes    
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > iou_threshhold)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    boxes_to_return = boxes[pick].astype("int")

    #convert back to [x,y,w,h] format
    return [[box[0],box[1],box[2]-box[0],box[3]-box[1]] for box in boxes_to_return]

def display_all_bounding_boxes(img,bounding_boxes):
    """
        Description: Displays final bounding boxes by looping through and drawing rectangles on image
    """

    draw_img = img.copy()
    for box in bounding_boxes:
        x,y,w,h = box
        resized_cropped_image = cv2.resize(draw_img[y:(y+h),x:(x+w)], (244,244), interpolation = cv2.INTER_AREA)
        cv2.rectangle(draw_img, (x,y), (x+w,y+h), (255,255,255), 4)
        
    cv2.imshow("Final Objects",draw_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_proposed_regions(img,limit_of_proposals= 2000):
    """
        Generate proposed regions using selective search and limits number of rectangles returned
    """
    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(img)
    selective_search.switchToSelectiveSearchFast() # lower likelihood of getting correct bounding boxes but quicker
    all_proposed_rectangles = selective_search.process() 
    limited_proposed_recentagles = [v for i,v in enumerate(all_proposed_rectangles) if i < limit_of_proposals]
    return limited_proposed_recentagles

if __name__ == "__main__":
    images_to_predict = get_list_of_images_not_predicted()
    model = models.load_model(config["ModelPath"])
    for input_img_file in images_to_predict:
        img = cv2.imread(input_img_file)

        rects = generate_proposed_regions(img)

        cropped_images = [get_resized_img_to_model(img,box,resize_dimensions=config["ClassifierInput"]) for box in rects]
        prediction_ready_images = np.array([applications.vgg16.preprocess_input(img) for img in cropped_images])

        predictions = model.predict(prediction_ready_images) # Takes the longest time, if there is anyway to optimise this then that will be your best bet

        bounding_boxes_above_threshold = [rects[i] for i,pred in enumerate(predictions) if pred[1] > config["ImageThreshold"]]

        final_bounding_box_objects = non_max_suppression_fast(bounding_boxes_above_threshold, config["IouThreshold"])

        file_ext = os.path.splitext(input_img_file)[-1].lower()
        base_file_name = os.path.basename(input_img_file)[:-len(file_ext)]

        predictions_data_dump = {
            "ImageName" :  base_file_name
            ,"FileExtension" : file_ext
            ,"CV2Image" : img
            ,"Rectangles" : rects
            ,"Predictions" : predictions
            ,"FinalBoundingBoxes" : final_bounding_box_objects
        }
        
        pickle.dump(predictions_data_dump,open( config["OutputSaveLocation"]+os.sep+predictions_data_dump["ImageName"]+".p", "wb" ))

        display_all_bounding_boxes(img,final_bounding_box_objects)

        _, foreground_max = predictions.max(axis=0)
        max_index = [i for i,v in enumerate(predictions) if v[1] == foreground_max][0]

        foreground_predictions_with_index = [[i,v[1]] for i,v in enumerate(list(predictions))]
        sorted_foreground_predictions_with_index = sorted(foreground_predictions_with_index, reverse=True,key=lambda x: x[1])

        display_selected_regions(img,predictions,rects,config["ImageThreshold"])

        display_heat_map_of_object(img,predictions,rects,config["ImageThreshold"])
        
    print("Finishing")

     