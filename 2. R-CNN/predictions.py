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
        1) Must have TF imported otherwise it will not run because of the lambda layer
"""

config = {
    "InputFIle" : "1. Data Gen\\2. Data for Predictions\\WIN_20210426_12_27_45_Pro.jpg" #"1. Data Gen\\2. Data for Predictions\\WIN_20210423_11_21_30_Pro.jpg"
    ,"ModelPath" : "2. Models\\vggtrained.h5"
    ,"ClassifierInput" : (224,224)
    , "ModelInput" : (224,224)
    ,"OutputSaveLocation" : "1. Data Gen\\3. Data From Predictions"
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



if __name__ == "__main__":
    model = models.load_model(config["ModelPath"])
    img = cv2.imread(config["InputFIle"])

    selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    selective_search.setBaseImage(img)
    selective_search.switchToSelectiveSearchFast() # lower likelihood of getting correct bounding boxes but quicker
    rects = selective_search.process()

    cropped_images = [get_resized_img_to_model(img,box,resize_dimensions=config["ClassifierInput"]) for box in rects]
    prediction_ready_images = np.array([applications.vgg16.preprocess_input(img) for img in cropped_images])
    predictions = model.predict(prediction_ready_images) # Takes the longest time!

    #predictions = pickle.load(open("1. Data Gen\\3. Data From Predictions\\WIN_20210426_12_27_45_Pro.p","rb"))["Predictions"]
    file_ext = os.path.splitext(config["InputFIle"])[-1].lower()
    base_file_name = os.path.basename(config["InputFIle"])[:-len(file_ext)]


    predictions_data_dump = {
        "ImageName" :  base_file_name
        ,"FileExtension" : file_ext
        ,"CV2Image" : img
        ,"Rectangles" : rects
        ,"Predictions" : predictions
    }
    #pickle.dump(predictions_data_dump,open( config["OutputSaveLocation"]+os.sep+predictions_data_dump["ImageName"]+".p", "wb" ))

    _, foreground_max = predictions.max(axis=0)
    max_index = [i for i,v in enumerate(predictions) if v[1] == foreground_max][0]

    foreground_predictions_with_index = [[i,v[1]] for i,v in enumerate(list(predictions))]
    sorted_foreground_predictions_with_index = sorted(foreground_predictions_with_index, reverse=True,key=lambda x: x[1])

    for index_and_prediction in sorted_foreground_predictions_with_index[:100]:
        index, prediction_result = index_and_prediction
        shown_image = img.copy()
        x,y,w,h = rects[index]
        cv2.rectangle(shown_image,(x,y),(x+w,y+h),(255,255,255),1)
        print(f"[{index}]={prediction_result}")
        cv2.imshow(f"Image",shown_image)
        cv2.waitKey(1000)

    cv2.destroyAllWindows()
    
    height, width, _ = img.shape
    greyscale_mask = np.zeros((height, width))
    increase = int(255/25)
    for index_and_prediction in sorted_foreground_predictions_with_index[:25]:
        index, prediction_result = index_and_prediction
        x,y,w,h = rects[index]
        rectangular_box_mask = greyscale_mask.copy()
        cv2.rectangle(rectangular_box_mask,(x,y),(x+w,y+h),(255),1)
        result = cv2.bitwise_and(greyscale_mask,rectangular_box_mask,mask = rectangular_box_mask)
        cv2.imshow("masked result",result)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        

    """
    already_predicted_index = [0] * len(rects)
    random_search_step = 25
    fore_max_threshold = 0.7
    
    predictions = np.array([])
    search_start_index = 0
    
    for loop_num in range(1,int(len(cropped_images)/random_search_step)):
        search_finish_index = random_search_step*loop_num
        if loop_num == 1:
            predictions = model.predict(prediction_ready_images[search_start_index:search_finish_index])
        else:   
            predictions = np.append(predictions, model.predict(prediction_ready_images[search_start_index:search_finish_index]),axis=0)

        _, foreground_max = predictions.max(axis=0)
        max_index = [i for i,v in enumerate(predictions) if v[1] == foreground_max][0]

        for i in range(search_start_index,search_finish_index):
            print(f"Index [{i}] with likelihood of {predictions[i][1]}")

        already_predicted_index = [int(i <= search_finish_index and i>= search_start_index) for i,v in enumerate(already_predicted_index)]
        print(f"loop [{loop_num}] [{search_start_index}-{search_finish_index}] max foreground likelihood is {foreground_max}")

        if foreground_max > fore_max_threshold:
            print("Found suitable candidate")
        search_start_index = search_finish_index
    """
    print("finish")
    #cropped_images = [get_resized_img_to_model(img,box,resize_dimensions=config["ClassifierInput"]) for box in rects]
    #prediction_ready_images = np.array([applications.vgg16.preprocess_input(img) for img in cropped_images])
    #predictions = model.predict(prediction_ready_images[:100]) # Only do 100 it is very time intensive to run the classifier
 
    #chosen_box, fore_max, previous_predictions_list, previous_boxes_list = get_nearest_box(img, predictions,rects)
    #print(f"{chosen_box} {fore_max}")

    #chosen_box, fore_max, previous_predictions_list, previous_boxes_list = get_nearest_box(img, previous_predictions_list,previous_boxes_list)
    #print(f"{chosen_box} {fore_max}")

    """
    cv2.rectangle(img, (chosen_box[0],chosen_box[1]), (chosen_box[0]+chosen_box[2],chosen_box[1]+chosen_box[3]), (255,255,255), 1)
    cv2.imshow("Final Result",img)
    cv2.waitKey(0)
    bounding_box_result = []
    """
    print("Finishing")

     