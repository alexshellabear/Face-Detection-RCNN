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
"""

config = {
    "ModelPath" : "2. Models\\vggtrained 27 April 21.h5"
    ,"ClassifierInput" : (224,224)
    ,"ModelInput" : (224,224)
    ,"OutputSaveLocation" : "1. Data Gen\\3. Data From Predictions"
    ,"InputImageFiles" : "1. Data Gen\\2. Data for Predictions"
    ,"ValidImageFileExtensions" : [".jpg",".png"]
    ,"ValidPredictionFileExtensions" : [".p"]
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
    print(box)
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
def cv2_resize(base_image,label):
    """
        Wrapper for cv2.resize for readability
    """
    return cv2.resize(base_image[label["Box"]['y1']:label["Box"]['y2'],label["Box"]['x1']:label["Box"]['x2']], config["ModelInput"], interpolation = cv2.INTER_AREA)
    
if __name__ == "__main__":
    
    images_to_predict = get_list_of_images_not_predicted()
    #images_to_predict = ["1. Data Gen\\2. Data for Predictions\\WIN_20210427_16_04_34_Pro.jpg"]
    for input_img_file in images_to_predict:
        model = models.load_model(config["ModelPath"])
        img = cv2.imread(input_img_file)

        selective_search = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
        selective_search.setBaseImage(img)
        selective_search.switchToSelectiveSearchFast() # lower likelihood of getting correct bounding boxes but quicker
        rects = selective_search.process()

        cropped_images = [get_resized_img_to_model(img,box,resize_dimensions=config["ClassifierInput"]) for box in rects]
        prediction_ready_images = np.array([applications.vgg16.preprocess_input(img) for img in cropped_images])

        #predictions = pickle.load(open("1. Data Gen\\3. Data From Predictions\\WIN_20210427_16_04_34_Pro.p","rb"))["Predictions"]
        predictions = model.predict(prediction_ready_images) # Takes the longest time!

        file_ext = os.path.splitext(input_img_file)[-1].lower()
        base_file_name = os.path.basename(input_img_file)[:-len(file_ext)]

        predictions_data_dump = {
            "ImageName" :  base_file_name
            ,"FileExtension" : file_ext
            ,"CV2Image" : img
            ,"Rectangles" : rects
            ,"Predictions" : predictions
        }
        
        pickle.dump(predictions_data_dump,open( config["OutputSaveLocation"]+os.sep+predictions_data_dump["ImageName"]+".p", "wb" ))

        _, foreground_max = predictions.max(axis=0)
        max_index = [i for i,v in enumerate(predictions) if v[1] == foreground_max][0]

        foreground_predictions_with_index = [[i,v[1]] for i,v in enumerate(list(predictions))]
        sorted_foreground_predictions_with_index = sorted(foreground_predictions_with_index, reverse=True,key=lambda x: x[1])


        # Loop through top 25 selections and display them for 2 seconds each
        #top_25_bounding_box_masks = [create_mask(img,rects[i],pixel_colour) for i,v in sorted_foreground_predictions_with_index[:25]]
        exclude_images = [
                "1. Data Gen\\1. Data\\WIN_20210426_12_27_45_Pro.jpg"
                ,"1. Data Gen\\1. Data\\WIN_20210425_17_03_57_Pro.jpg"
                ,"1. Data Gen\\1. Data\\WIN_20210425_17_03_54_Pro.jpg"
                ,"1. Data Gen\\1. Data\\WIN_20210425_17_03_46_Pro.jpg"
                ,"1. Data Gen\\1. Data\\WIN_20210425_17_03_48_Pro.jpg"
                ,"1. Data Gen\\1. Data\\WIN_20210423_11_21_30_Pro.jpg"
            ] # TODO delete later

        drawn_img = img.copy()
        cv2.imshow("Top 25",drawn_img)
        cv2.waitKey(10000)
        for index, prediction in sorted_foreground_predictions_with_index[:100]:
            x,y,w,h = rects[index]
            resized_cropped_image = cv2_resize(base_image,label)
            cv2.rectangle(base_image, (x,y), (x+w,y+h), (255,255,255), 4) 
            cv2.putText(base_image,f"Prob of My Face = {prediction}", 
                (100,100), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (255,255,255),
                2)
            
            cv2.imshow("BaseImage",base_image)
            cv2.imshow("ResizedImage",resized_cropped_image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

        pixel_colour = int(255/25)
        bounding_box_masks = [create_mask(img,rects[i],pixel_colour) for i,v in sorted_foreground_predictions_with_index[:25]]
        covered_area = sum(bounding_box_masks)
        cv2.imshow("Heat Map of Top 25 Search Areas",covered_area)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

        height, width, _ = img.shape
        covered_area = np.zeros((height, width),dtype=np.uint8)
        for i,mask in enumerate(bounding_box_masks):
            index, _ = sorted_foreground_predictions_with_index[i]
            x,y,w,h = rects[index]
            covered_area[y:y+h, x:x+w] = covered_area[y:y+h, x:x+w] + pixel_colour
        
        for index_and_prediction in sorted_foreground_predictions_with_index[:100]:
            index, prediction_result = index_and_prediction
            shown_image = img.copy()
            x,y,w,h = rects[index]
            cv2.rectangle(shown_image,(x,y),(x+w,y+h),(255,255,255),1)
            print(f"[{index}]={prediction_result}")
            cv2.imshow(f"Heat Map of top",shown_image)
            cv2.waitKey(1000)

        cv2.destroyAllWindows()
        
    print("Finishing")

     