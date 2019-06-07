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
import os

xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report

# Path to directory containing images to be verified by model
# Images must be in a subdirectory of this path
directory = "/path/to/demo"

# Get path to model. model_name should be name of model used
model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'horizon.hdf5'

model_path = os.path.join(model_dir, model_name)

# Dimensions of input image. Image will be scaled to these dimensions
# Must be the same input shape as specified during training
shape=(214,214)
resize_shape=(shape[0],shape[1],3)

# Use ImageDataGenerator to load images
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

# Count images in subdirectories
names = val_gen.filenames
step = len(names)

# Load the model
cifar_model = keras.models.load_model(model_path)

num_classes = 2

# Print length of names for debugging purposes
print(len(names))

# Predict classes of the images
predicted_classes = cifar_model.predict_generator(val_gen, steps=step)

# Print predictions
print("Printing predictions for horizon model: [true | false]")
for x in range(len(names)):
    print("File " + names[x] + " result: " + str(predicted_classes[x]))

exit(0)
