import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report

# Path to validation/demo directory
# The images need to be in a subdirectory of directory given by the path below
directory = "/path/to/demo/"

# Get import path for model. Model path should point to the model you wish to test
model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'wells.hdf5'

model_path = os.path.join(model_dir, model_name)

# Input shape should match input shape of the trained model
# An earlier well model was set to (224, 224) due to a typo
shape=(214,214)
resize_shape=(shape[0],shape[1],3)

# Use ImageDataGenerator to load the images
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

# Find amount of image in the subfolders
names = val_gen.filenames
step = len(names)

# Load the model
cifar_model = keras.models.load_model(model_path)

num_classes = 2

print(len(names))

# Predict classes for images
# For each image, returns [prob true, prob false]
predicted_classes = cifar_model.predict_generator(val_gen, steps=step)

# Print the predictions
print("Printing predictions for horizon model: [true | false]")
for x in range(len(names)):
    print("File " + names[x] + " result: " + str(predicted_classes[x]))

exit(0)
