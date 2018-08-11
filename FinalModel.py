import matplotlib
matplotlib.use("Agg")

# import the necessary packages

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import itertools
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
import matplotlib.pyplot as plt
import argparse
import random
import cv2
import os
from keras.models import Sequential
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

%matplotlib inline

def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=contains)

def list_files(basePath, validExts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"), contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()

            # check to see if the file is an image and should be processed
            if ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename).replace(" ", "\\ ")
                yield imagePath




#def build(width, height, depth, classes):
        # initialize the model



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5
KTF.set_session(tf.Session(config=config))


height = 160
width = 160
depth = 3

train_path = '/data/v3/training'
validation_path = '/data/v3/validation'
test_path = '/data/v3/test'

test_path = sorted(list(list_images(test_path)))
image_path = sorted(list(list_images(train_path)))
validation_path = sorted(list_images(validation_path))
random.seed(42)

random.shuffle(image_path)
random.shuffle(validation_path)
random.shuffle(test_path)

data = []
labels = []

for imagePath in image_path:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        data.append(image)
    
        label = imagePath.split(os.path.sep)[-2]
    
        if label == "AFRAID":
            label = 0
        
        elif label == "ANGRY":
            label = 1
    
        elif label == "DISGUST" :
            label = 2
    
        elif label == "HAPPY" :
            label = 3
    
        elif label == "NEUTRAL": 
            label = 4
    
        elif label == "SAD" :
            label = 5
    
        elif label == "SURPRISED":
            label = 6
        else:
            label = None
    
        labels.append(label)

vdata=[]
vlabels=[]

for imagePath in validation_path:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        vdata.append(image)
    
        vlabel = imagePath.split(os.path.sep)[-2]
    
        if vlabel == "AFRAID":
            vlabel = 0
        
        elif vlabel == "ANGRY":
            vlabel = 1
    
        elif vlabel == "DISGUST" :
            vlabel = 2
    
        elif vlabel == "HAPPY" :
            vlabel = 3
    
        elif vlabel == "NEUTRAL": 
            vlabel = 4
    
        elif vlabel == "SAD" :
            vlabel = 5
    
        elif vlabel == "SURPRISED":
            vlabel = 6
        
    
        vlabels.append(vlabel)

print(len(image_path))
print (len(validation_path))

data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
vdata = np.array(vdata,dtype="float")/ 255.0
vlabels = np.array(vlabels)

trainX = list (data)
print(len(data))
trainY = list (labels)
print(len(labels))
valX = list (vdata)
print(len(vdata))
valY = list (vlabels)
print(len(vlabels))

trainY = np_utils.to_categorical(trainY, num_classes=7)
valY = np_utils.to_categorical(valY, num_classes = 7)

#trainX = trainX.reshape(trainX.shape[0], 3, 28, 128).astype('float32')
#testX = testX.reshape(testX.shape[0], 3, 128, 128).astype('float32')

print(trainY.shape)
print(valY.shape)

EPOCHS = 50
INIT_LR = 1e-3
BS = 128
#valid_batch = 15



classes = 7

aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
horizontal_flip=True, fill_mode="nearest")

model = Sequential()
inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
if K.image_data_format() == "channels_first":
    inputShape = (depth, height, width)

            # first set of CONV => RELU => POOL layers
model.add(Conv2D(32, (3, 3), padding="same",input_shape=inputShape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# second set of CONV => RELU => POOL layers
model.add(Conv2D(64, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))

#temporary layer?
model.add(Conv2D(128, (4, 4), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.7))



# first (and only) set of FC => RELU layers
model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))
model.add(Dropout(0.45))

# softmax classifier
model.add(Dense(classes))
model.add(Activation("softmax"))

# initialize the model
print("[INFO] compiling model...")

opt = Adam(lr=0.0006)
model.compile(loss="categorical_crossentropy", optimizer=opt,
metrics=["accuracy"])

#aug.fit(np.array(trainX))

early= EarlyStopping(monitor= 'val_acc', patience = 3)

print("[INFO] training network...")
#model.fit(np.array(trainX), np.array(trainY), batch_size=BS,
#          validation_data=(np.array(valX), np.array(valY)),
#    epochs=EPOCHS, verbose=1)

#checkpoint_callback= ModelCheckpoint('models/model_checkpoint.h5', monitor = 'val_acc', verbose = 1, save_best_only= True, save_weights_only= False, mode = 'auto', period = 1)

history = model.fit_generator(aug.flow(np.array(trainX), np.array(trainY), batch_size=BS),
                    validation_data=(np.array(valX), np.array(valY)), 
                    steps_per_epoch=len(trainX) // 32,
                    epochs=EPOCHS, verbose=1)

tdata=[]
tlabels=[]


for imagePath in test_path:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (width, height))
        image = img_to_array(image)
        tdata.append(image)
    
        tlabel = imagePath.split(os.path.sep)[-2]
    
        if tlabel == "AFRAID":
            tlabel = 0
        
        elif tlabel == "ANGRY":
            tlabel = 1
    
        elif tlabel == "DISGUST" :
            tlabel = 2
    
        elif tlabel == "HAPPY" :
            tlabel = 3
    
        elif tlabel == "NEUTRAL": 
            tlabel = 4
    
        elif tlabel == "SAD" :
            tlabel = 5
    
        elif tlabel == "SURPRISED":
            tlabel = 6
        
    
        tlabels.append(tlabel)

tdata = np.array(tdata,dtype="float")/ 255.0
tlabels = np.array(tlabels)

testX = list(tdata)
print(len(tdata))
testY = list (tlabels)
print(len(tlabels))

testY = np_utils.to_categorical(testY, num_classes = 7)

score = model.evaluate(np.array(testX),np.array(testY))
print(score)

model.summary()
