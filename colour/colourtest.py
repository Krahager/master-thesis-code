import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import os

directory = "/home/krahager/PyUiTestResults/ViewportColorTest"

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'background_color.hdf5'

shape = (214, 214)
resize_shape = (shape[0], shape[1], 3)

num_classes = 2

datagen = ImageDataGenerator(validation_split=0.2, samplewise_std_normalization=False,
                              # width_shift_range=0.2,
                              # height_shift_range=0.2,
                              # zoom_range=1.1,
                              # rotation_range=45
                             )
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, model_name)
rms_model = keras.models.load_model(model_path)


pred_gen = datagen.flow_from_directory(
        directory,
        classes=['other', 'white'],
        target_size=shape,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=False,
        batch_size=1,
        subset='validation'
        )

names = pred_gen.filenames
step = len(names)

# Display classification report
predicted_classes = rms_model.predict_generator(pred_gen, steps=step)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
#predicted_classes = np.rint(predicted_classes)
y_val = pred_gen.classes
print(y_val)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(confusion_matrix(y_val, predicted_classes))
print(classification_report(y_val, predicted_classes, target_names=target_names))
