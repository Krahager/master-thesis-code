import tensorflow
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
#import msdexternal.models.msdmodel as M
import os

xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report

directory = "/home/krahager/PyUiTestResults/MultiDetectionTest/demo"

model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'wells.15-0.13.hdf5'

model_path = os.path.join(model_dir, model_name)

shape=(64,64)
resize_shape=(shape[0],shape[1],3)

#from keras.datasets import cifar10
datagen = ImageDataGenerator(validation_split=None)
# train_gen = datagen.flow_from_directory(
#         directory,
#         target_size=shape,
#         color_mode='rgb',
#         class_mode="sparse",
#         shuffle=True,
#         batch_size=64,
#         subset='training'
#         )
val_gen = datagen.flow_from_directory(
        directory,
        target_size=shape,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=False,
        batch_size=1,
        subset=None
        )

names = val_gen.filenames
step = len(names)

cifar_model = keras.models.load_model(model_path)

num_classes = 2
# x, y = izip(*(val_gen[i] for i in xrange(len(val_gen))))
# x_val, y_val = np.vstack(x), np.vstack(y)
#train_X, train_Y = train_gen
#test_X, test_Y = val_gen

print(len(names))

# Display classification report
predicted_classes = cifar_model.predict_generator(val_gen, steps=step)
# print(predicted_classes)

print("Printing predictions for horizon model: [true | false]")
for x in range(len(names)):
    print("File " + names[x] + " result: " + str(predicted_classes[x]))

# predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
# target_names = ["Class {}".format(i) for i in range(num_classes)]
# print(classification_report(y_val, predicted_classes, target_names=target_names))
