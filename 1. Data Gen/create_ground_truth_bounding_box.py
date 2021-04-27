import cv2
import os

"""
    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Purpose:
        Create ground truth bounding boxes for foreground and background labeling (binary)

    User Guide:
        Click the first point where you would like the bounding box to start
        You should be able to see a bounding box move around as you select the second point
        Click the second point and now a bounding box has been determined

        You now have the option of fine tuning everything by using the following commands
        Up arrow: Moves the top edge of the bounding box up
        Left arrow: Moves the left edge of the bounding box to the left
        etc... same as down and right keys

        if tab is pressed once, not held, then this will make it go to the opposite direction

    Assumptions:
        1) Only images
        2) No duplicate names
        
    Lessons Learnt
        1) CV2 has a call back function for the window for on click events
        2) How to detect combinations of keys with cv2
            https://stackoverflow.com/questions/58408096/python-how-to-detect-key-combinations-with-opencv-waitkey-library
        3) How to detect shift key press using waitKey in cv2
            https://www.reddit.com/r/learnpython/comments/6vapuy/how_do_i_detect_key_presses_and_mouse_clicks_in/
            https://stackoverflow.com/questions/54846803/how-to-detect-shift-or-ctrl-with-arrow-keys-when-using-the-waitkeyex

    TODO When using buttons the last known position of the mouse is off the screen. 
"""

config = {
    "ImagePath" : "1. Data Gen\\1. Data\\test_cap.PNG"
    ,"DataDirectory" : "1. Data Gen\\1. Data" 
    ,"WindowName" : "Select Bounding Box"
    ,"PathToTutorial" : "1. Data Gen\\Ground Truth Tutorial.png"
    ,"UpKey" : 2490368
    ,"LeftKey" : 2424832
    ,"DownKey" : 2621440
    ,"RightKey" : 2555904
    ,"TabKey" : 9
    ,"PathToData" : "1. Data Gen\\1. Data"
    ,"ValidImageFileExtensions" : [".jpg",".png"]
    ,"ValidLabelFileExtensions" : ['.csv']
}

X_start = 0
Y_start = 0
cur_mouse_X = 0
cur_mouse_y = 0
left_mouse_down = False
first_point_selected = False
box_drawn = False 
box_start_x = 0
box_start_y = 0
box_finish_x = 0
box_finish_y = 0
shift_key_down = False


def draw_and_save_label(mouseX,mouseY,box_start_x_increase=0,box_start_y_increase=0,box_finish_x_increase=0,box_finish_y_increase=0):
    global X_start,Y_start,left_mouse_down,first_point_selected,img,drawn_img,box_drawn,cur_mouse_X,cur_mouse_y, box_start_x, box_start_y, box_finish_x, box_finish_y, shift_key_down

    if box_start_x_increase == 0 and box_drawn == False:
        box_start_x = X_start if X_start < mouseX else mouseX
    else:
        if shift_key_down:
            box_start_x -= box_start_x_increase
        else:
            box_start_x += box_start_x_increase

    if box_start_y_increase == 0 and box_drawn == False:
        box_start_y = Y_start if Y_start < mouseY else mouseY 
    else:
        if shift_key_down:
            box_start_y -= box_start_y_increase
        else:
            box_start_y += box_start_y_increase

    if box_finish_x_increase == 0 and box_drawn == False:
        box_finish_x = X_start if X_start > mouseX else mouseX
    else:
        if shift_key_down:
            box_finish_x -= box_finish_x_increase
        else:
            box_finish_x += box_finish_x_increase

    if box_finish_y_increase == 0 and box_drawn == False:
        box_finish_y = Y_start if Y_start > mouseY else mouseY
    else:
        if shift_key_down:
            box_finish_y -= box_finish_y_increase
        else:
            box_finish_y += box_finish_y_increase

    drawn_img = img.copy() # Reset image
    if box_drawn == False:
        cv2.rectangle(drawn_img, (X_start,Y_start), (mouseX,mouseY), (255,255,255), 1) # Draw
    else:
        cv2.rectangle(drawn_img, (box_start_x,box_start_y), (box_finish_x,box_finish_y), (255,255,255), 1)
    csv_name = f"{config['ImagePath'][:-4]}.csv"

    csv_string = f"X_start\tY_start\tX_finish\tY_finish\n{box_start_x}\t{box_start_y}\t{box_finish_x}\t{box_finish_y}"
    with open(csv_name,"w") as file_obj:
        file_obj.write(csv_string)
        file_obj.close()

def draw_and_save_callback(event,mouseX,mouseY,flags,param):
    global X_start,Y_start,left_mouse_down,first_point_selected,img,drawn_img,box_drawn,cur_mouse_X,cur_mouse_y,shift_key_down
    cur_mouse_X = mouseX
    cur_mouse_y = mouseY
    if event == cv2.EVENT_LBUTTONDOWN and left_mouse_down == False:
        left_mouse_down = True
        print(f"Left Button Down, {mouseX}:{mouseY} , {flags}, {param}, first_point_selected = {first_point_selected}, left_mouse_down = {left_mouse_down}")
        
    elif event == cv2.EVENT_LBUTTONUP and left_mouse_down == True:
        left_mouse_down = False
        print(f"Left Button Up, {mouseX}:{mouseY} , {flags}, {param}  first_point_selected = {first_point_selected}, left_mouse_down = {left_mouse_down}")
        
        if first_point_selected == False:
            first_point_selected = True
            X_start = mouseX
            Y_start = mouseY
        else:
            first_point_selected = False
            draw_and_save_label(mouseX,mouseY)
            
            box_drawn = True
    elif event == "key button pressed":
        if param == config["TabKey"]:
            if shift_key_down == True:
                shift_key_down = False
            else:
                shift_key_down = True

        if box_drawn == True:
            if param == config["UpKey"]:
                draw_and_save_label(mouseX,mouseY,box_start_y_increase=-1)
            elif param == config["LeftKey"]:
                draw_and_save_label(mouseX,mouseY,box_start_x_increase=-1)
            elif param == config["DownKey"]:
                draw_and_save_label(mouseX,mouseY,box_finish_y_increase=1)
            elif param == config["RightKey"]:
                draw_and_save_label(mouseX,mouseY,box_finish_x_increase=1)
    elif first_point_selected == True:
        print(f"DRAW {event}, {mouseX}:{mouseY} , {flags}, {param}  {flags}, {param}  first_point_selected = {first_point_selected}, left_mouse_down = {left_mouse_down}")
        drawn_img = img.copy() # Reset image
        cv2.rectangle(drawn_img,(X_start,Y_start),(mouseX,mouseY),(255,255,255),1)
    else:
        print(f"{event}, {mouseX}:{mouseY} , {flags}, {param}  {flags}, {param}  first_point_selected = {first_point_selected}, left_mouse_down = {left_mouse_down}")

def get_list_of_unlabeled_data():
    """
        Description: Loop through data and find image files & corresponding csv files. For those that HAVE corresponding file extensions remove from list
        Assumption 1: Those images which have a ground truth will have the same file name but with a csv file extension
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

    list_of_imgs_with_no_label = []
    for img_full_file_name in list_of_imgs:
        img_file_name = os.path.splitext(img_full_file_name)[0].lower()
        if not img_file_name in [os.path.splitext(label_full_file_name)[0].lower() for label_full_file_name in list_of_img_labels]:
            list_of_imgs_with_no_label.append(img_full_file_name)

    return list_of_imgs_with_no_label 

def display_tutorial():
    """
        Description: Displays a tutorial image which shows how to use the labelling tool until any key is pressed.
    """
    tutorial_image = cv2.imread(config["PathToTutorial"])
    cv2.imshow(config["WindowName"])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print('starting...')

    display_tutorial()

    list_of_imgs = get_list_of_unlabeled_data()
    
    for img_path in list_of_imgs:
        first_point_selected = False
        box_drawn = False
        config["ImagePath"] = img_path
        img = cv2.imread(img_path)
        cv2.namedWindow(config["WindowName"])
        cv2.setMouseCallback(config["WindowName"],draw_and_save_callback)
        drawn_img = img.copy()

        while(1):
            
            cv2.imshow(config["WindowName"],drawn_img)
            pressed_button = cv2.waitKeyEx(20)
            if pressed_button == 27:
                cv2.destroyAllWindows()
                break  
            elif pressed_button != -1:
                draw_and_save_callback("key button pressed",cur_mouse_X,cur_mouse_y,0,pressed_button)
        
        