import tensorflow
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D
from keras.layers import ELU
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras import regularizers
import os

from keras.preprocessing.image import ImageDataGenerator

xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report, confusion_matrix

#Replace with path to wells dataset. Dataset should contain subfolders 'true' and 'false'
directory = "/path/to/wells"

# Where the model(s) should be saved
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_wells_model.h5'

# Input shape (generator will resize input images to these dimensions)
# Should NOT be below 214x214 with 5 Convolution blocks
shape=(214,214)
resize_shape=(shape[0],shape[1],3)

batch_size = 16
epochs = 40
num_classes = 2

datagen = ImageDataGenerator(validation_split=0.2, samplewise_std_normalization=False,
                              # width_shift_range=0.2,
                              # height_shift_range=0.2,
                              # zoom_range=1.1,
                              # rotation_range=45
                             )
train_gen = datagen.flow_from_directory(
        directory,
        classes=['true', 'false'],
        target_size=shape,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
        subset='training'
        )
val_gen = datagen.flow_from_directory(
        directory,
        classes=['true', 'false'],
        target_size=shape,
        color_mode='rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size=batch_size,
        subset='validation'
        )

#batch_size = 64

#setup model
rms_model = Sequential()

# Block 1
rms_model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same',
                     input_shape=resize_shape, kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(Conv2D(32, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(MaxPooling2D((2, 2), padding='same'))

# Block 2
rms_model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same',
                     input_shape=resize_shape, kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(Conv2D(64, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(MaxPooling2D((2, 2), padding='same'))

# Block 3
rms_model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same',
                     input_shape=resize_shape, kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(Conv2D(128, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(MaxPooling2D((2, 2), padding='same'))

# Block 4
rms_model.add(Conv2D(256, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(Conv2D(256, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Block 5
rms_model.add(Conv2D(512, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(Conv2D(512, (3, 3), activation='linear', padding='same', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(BatchNormalization())
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# Flatten to 1D array and classify
rms_model.add(Flatten())
rms_model.add(Dense(2048, activation='linear', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(LeakyReLU(alpha=0.01))
rms_model.add(Dropout(0.7))
rms_model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
rms_model.add(Dropout(0.7))
rms_model.add(Dense(num_classes, activation='softmax'))

print("Final layers added")

# Compile the model above
rms_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.00001, clipnorm=1.),
                  metrics=['accuracy'])

rms_model.summary()


# Checkpoint for saving the model every time the loss improves
filepath = os.path.join(save_dir, "wells_.{epoch:02d}-{val_loss:.2f}.hdf5")
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

# Tensorboard callback. Add to end of callbacks list in fit_generator below if uncommented
#tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

# This is where the training happens. 
# Without data augmentation, batch_size * steps_per_epoch should equal the size of the training set
train_data = rms_model.fit_generator(
        train_gen,
        callbacks=[checkpoint],
        steps_per_epoch=(84),
        epochs=epochs,
        validation_data=val_gen,
        validation_steps=21)

print(rms_model.get_weights())

# Plot training data
accuracy = train_data.history['acc']
val_accuracy = train_data.history['val_acc']
loss = train_data.history['loss']
val_loss = train_data.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
rms_model.save(model_path)
print('Saved trained model at %s ' % model_path)

# generator for classification report
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

names = pred_gen.filenames
step = len(names)

# Display classification report and confusion matrix
predicted_classes = rms_model.predict_generator(pred_gen, steps=step)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
y_val = pred_gen.classes
print(y_val)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(confusion_matrix(y_val, predicted_classes))
print(classification_report(y_val, predicted_classes, target_names=target_names))
