# Project Name: #
    Alex Face Detector
# Description: #
    This project is a custom R-CNN to develop bounding box object detection. 

    There are 4 key components
    1) Generate ground truths: Label the images
    2) Generate a dataset: Take ground truth labels and form a dataset from it
    3) Transfer learning: Use pre-existing weights of VGG16 imagenet and customise it to this dataset
    4) Predict: Use the trained model to make predictions on incoming images

# Table of contents: #


# Installation: #
    1) Make sure you have python installed, use the following link https://www.python.org/downloads/
    2) How to get started with python https://www.python.org/about/gettingstarted/
    3) How to set up your requirements with requirements.txt https://stackoverflow.com/questions/7225900/how-can-i-install-packages-using-pip-according-to-the-requirements-txt-file-from
    
    Okay you're now set with the right environment let's get this show on the road!

# Creating your own face detector: #
    Make your current working directory when running scripts the same as the one readme.md (this file) is stored in
        Psst you can check using os.getcwd()
    1) Take a bunch of photos of your face and drop them into the path "1. Data Gen\1. Data"
    2) Run the python script "1. Data Gen\create_ground_truth_bounding_box.py" this is a labelling tool that wcan enable you to label data.
        Assumption 1: Every image name is unique



