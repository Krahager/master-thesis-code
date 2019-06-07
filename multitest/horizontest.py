import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

# Path to horizon training/validation set
directory = "/path/to/horizon"

# Name and dir of the model to be loaded
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'seismic_cn.33-3.38.hdf5'

model_path = os.path.join(save_dir, model_name)

# Input shape, image will be scaled to this
# Must match the shape that was specified during training
shape = (214, 214)
resize_shape = (shape[0], shape[1], 3)

num_classes = 2

datagen = ImageDataGenerator(validation_split=0.2, samplewise_std_normalization=False,
                              # width_shift_range=0.2,
                              # height_shift_range=0.2,
                              # zoom_range=1.1,
                              # rotation_range=45
                             )


# Load model
rms_model = keras.models.load_model(model_path)

# Load data from subdirectories
pred_gen = datagen.flow_from_directory(
        directory,
        classes=['true', 'false'],
        target_size=shape,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=False,
        batch_size=1,
        subset='validation'
        )

# Count number of files in subdirectories
names = pred_gen.filenames
step = len(names)

# Display classification report and Confusion Matrix
predicted_classes = rms_model.predict_generator(pred_gen, steps=step)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
y_val = pred_gen.classes
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(confusion_matrix(y_val, predicted_classes))
print(classification_report(y_val, predicted_classes, target_names=target_names))
