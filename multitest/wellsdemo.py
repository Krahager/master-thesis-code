import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report

directory = "/home/krahager/PyUiTestResults/MultiDetectionTest/demo"

model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'wells_full_vgg.hdf5'

model_path = os.path.join(model_dir, model_name)

shape=(224,224)
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

print(len(names))

predicted_classes = cifar_model.predict_generator(val_gen, steps=step)

print("Printing predictions for horizon model: [true | false]")
for x in range(len(names)):
    print("File " + names[x] + " result: " + str(predicted_classes[x]))

exit(0)
