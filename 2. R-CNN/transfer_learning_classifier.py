from keras.layers import Dense
from keras import Model
from keras import models
from keras import optimizers
from keras import Sequential
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras import applications
from keras import utils
import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import cv2

"""

    Author: Alexander Shellabear
    Email: alexshellabear@gmail.com

    Purpose:
        Take dataset and train the classifier (in this case vgg16 to determine if the selective search areas are correct)

    Lessons Learnt
        1) Must add a Flatten layer between the models
        2) Model.fit can support ImageGenerators however you will need to pass it as both the x and y dataset. hence (x_train,y_train) would become train_datagen
"""

config = {
    "VGGModelPath" : "2. Models\\vggmodel.h5"
    ,"ModelInput" : (224,224)
    ,"EncodedLabels" : {
        "Background":0
        ,"Foreground" :1
        }
    ,"NumberOfCategories" : 2 
}

def count_number_of_this_labels(data_set_list:list,image_path_string:str):
    """
        Assumption: that data_set_list has dictionary elements with a 'ImagePath' element
    """
    return len([v for v in data_set_list if v["ImagePath"] == image_path_string])

def get_resized_images_for_data_set(data_set_labels: list):
    """
        Assumption 1: That data_set_list has dictionary elements with a 'ImagePath' and 'Box' elements
        Assumption 2: That 'Box' element is a dictionary with 'x1','y1','x2','y2' elements
        Output:
            A numpy array of all images referenced by the data_set_labels
    """
    data = []
    for label in data_set_labels:
        base_image = cv2.imread(label["ImagePath"])
        labeled_image = cv2.resize(base_image[label["Box"]['y1']:label["Box"]['y2'],label["Box"]['x1']:label["Box"]['x2']], config["ModelInput"], interpolation = cv2.INTER_AREA)
        data.append(labeled_image)
    return np.array(data)

def encode_labels(label_string):
    return [int(label == label_string) for label in config["EncodedLabels"]]

class MyLabelBinarizer(LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((Y, 1-Y))
        else:
            return Y
    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 0], threshold)
        else:
            return super().inverse_transform(Y, threshold)

def pre_process_data(x,y):
    x_preprocessed = applications.vgg16.preprocess_input(x) # Change to floats for each array
    y_preprocessed = utils.to_categorical(y,config["NumberOfCategories"]) # Change to one hot encoding, hence an array with the index of 1 corresponding to the label
    return x_preprocessed, y_preprocessed

if __name__ == "__main__":
    vggmodel = VGG16(weights='imagenet', include_top=False) # Load VGG16 for image classification, use include_top=False for retraining

    for layer in (vggmodel.layers)[:15]: # freeze the first 15 layers for retraining
        layer.trainable = False


    model = Sequential()
    model.add(layers.Lambda(lambda image: tf.image.resize(image, config["ModelInput"])))
    model.add(vggmodel) # Should have an output of 512 nodes
 
    model.add(layers.Flatten())

    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation='softmax'))
    
    all_background_labels = pickle.load(open("1. Data Gen\\1. Data\\background_labels.p","rb"))
    all_foreground_labels = pickle.load(open("1. Data Gen\\1. Data\\foreground_labels.p","rb"))

    background_dataset_labels = []
    for background in all_background_labels:
        img_path = background["ImagePath"] 
        if background["IOU"] == 0 and count_number_of_this_labels(background_dataset_labels,img_path) <  count_number_of_this_labels(all_foreground_labels,img_path):
            background_dataset_labels.append(background)
    
    X_data = get_resized_images_for_data_set(all_foreground_labels + background_dataset_labels)
    Y_data = np.array([config["EncodedLabels"][v["Label"]] for v in (all_foreground_labels + background_dataset_labels)])

    X_data,Y_data = pre_process_data(X_data,Y_data)

    x_train, x_test , y_train, y_test = train_test_split(X_data,Y_data,test_size=0.10)

    
    trdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    traindata = trdata.flow(x=x_train, y=y_train)
    tsdata = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=90)
    testdata = tsdata.flow(x=x_test, y=y_test)
    

    check_point = ModelCheckpoint(filepath="2. Models/weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5",
                                              monitor="val_acc",
                                              mode="max",
                                              save_best_only=True,
                                              )

    model.compile(loss='categorical_crossentropy',
                optimizer=optimizers.RMSprop(lr=2e-5),
                metrics=['accuracy']
                )

    #history = model.fit_generator(generator= traindata, steps_per_epoch= 10, epochs= 1000, validation_data= testdata, validation_steps=2) # Removed callbacks
    #history = model.fit(x_train,y_train, batch_size=32, epochs=10, verbose=1,validation_data=(x_test, y_test) )
    history = model.fit(traindata, batch_size=32, epochs=20, verbose=1,validation_data=testdata,callbacks=[check_point] ) # Known to work callbacks=[check_point]
    model.save("vggtrained.h5")
    print("finished...")