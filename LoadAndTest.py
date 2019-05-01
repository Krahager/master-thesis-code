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
xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report

directory = "/home/krahager/PyUiTestResults/ViewportColorTest"

#from keras.datasets import cifar10
datagen = ImageDataGenerator(validation_split=0.3)
train_gen = datagen.flow_from_directory(
        directory,
        target_size=(256,256),
        color_mode='rgb',
        class_mode="sparse",
        shuffle=True,
        batch_size=64,
        subset='training'
        )
val_gen = datagen.flow_from_directory(
        directory,
        target_size=(256,256),
        color_mode='rgb',
        class_mode="sparse",
        shuffle=True,
        batch_size=64,
        subset='validation'
        )

cifar_model = keras.models.load_model('/home/krahager/training_result_white_background.h5')

num_classes = 2
x, y = izip(*(val_gen[i] for i in xrange(len(val_gen))))
x_val, y_val = np.vstack(x), np.vstack(y)
#train_X, train_Y = train_gen
#test_X, test_Y = val_gen

# Display classification report
predicted_classes = cifar_model.predict(x_val)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_val, predicted_classes, target_names=target_names))
