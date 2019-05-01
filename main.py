import tensorflow
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AvgPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

from keras.preprocessing.image import ImageDataGenerator
#import msdexternal.models.msdmodel as M
xrange = range
izip = zip
imap = map

from sklearn.metrics import classification_report, confusion_matrix

directory = "/home/krahager/PyUiTestResults/ViewportColorTest"

#from keras.datasets import cifar10
datagen = ImageDataGenerator(validation_split=0.3, samplewise_std_normalization=True)
train_gen = datagen.flow_from_directory(
        directory,
        classes=['white', 'other'],
        target_size=(256,256),
        color_mode='rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size=16,
        subset='training'
        )
val_gen = datagen.flow_from_directory(
        directory,
        classes=['white', 'other'],
        target_size=(256,256),
        color_mode='rgb',
        class_mode="categorical",
        shuffle=True,
        batch_size=16,
        subset='validation'
        )

#batch_size = 64
batch_size = 16
epochs = 20
num_classes = 2

#setup model
#cifar_model = M.msdnet(3, (2, 8), 10, use_dropout=True, dropout=0.25) #3, (2,2), nClasses
rms_model = Sequential()
rms_model.add(Conv2D(4, kernel_size=(3, 3), strides=(1, 1), activation='linear', padding='same', input_shape=(256, 256, 3)))
rms_model.add(LeakyReLU(alpha=0.1))
rms_model.add(MaxPooling2D((2, 2), padding='same'))
rms_model.add(Dropout(0.7))
rms_model.add(Conv2D(4, (3, 3), activation='linear', padding='same'))
rms_model.add(LeakyReLU(alpha=0.1))
rms_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
rms_model.add(Dropout(0.6))
rms_model.add(Conv2D(4, (3, 3), activation='linear', padding='same'))
rms_model.add(LeakyReLU(alpha=0.1))
rms_model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
rms_model.add(Dropout(0.7))
rms_model.add(Flatten())
rms_model.add(Dense(1024, activation='tanh')) #tanh
# rms_model.add(LeakyReLU(alpha=0.1))
rms_model.add(Dropout(0.6))
rms_model.add(Dense(512, activation='tanh')) #sigmoid
# rms_model.add(LeakyReLU(alpha=0.1))
rms_model.add(Dropout(0.6))
rms_model.add(Dense(num_classes, activation='softmax'))

print("Final layers added")

#keras.losses.categorical_crossentropy
rms_model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

rms_model.summary()

print(rms_model.get_weights())

train_data = rms_model.fit_generator(
        train_gen,
        steps_per_epoch=30,
        epochs=30,
        validation_data=val_gen,
        validation_steps=16)

print(rms_model.get_weights())

#cifar_train = cifar_model.fit(x=train_X, y=train_label, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_label))

#test_eval = cifar_model.evaluate(test_X, test_Y_one_hot, verbose=0)

#print('Test loss:', test_eval[0])
#print('Test accuracy:', test_eval[1])

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

rms_model.save('/home/krahager/training_result_white_background_1.h5')

pred_gen = datagen.flow_from_directory(
        directory,
        classes=['white', 'other'],
        target_size=(256,256),
        color_mode='rgb',
        class_mode="categorical",
        shuffle=False,
        batch_size=1,
        subset='validation'
        )
# Display classification report
predicted_classes = rms_model.predict_generator(pred_gen, steps=254)
predicted_classes = np.argmax(np.round(predicted_classes), axis=1)
#predicted_classes = np.rint(predicted_classes)
y_val = pred_gen.classes
print(y_val)
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(confusion_matrix(y_val, predicted_classes))
print(classification_report(y_val, predicted_classes, target_names=target_names))

exit(0)
