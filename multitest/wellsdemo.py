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

layer_outputs_0 = [layer.output for layer in cifar_model.layers[:1]]

activation_model_0 = keras.models.Model(inputs=cifar_model.input, outputs=layer_outputs_1)

models = [activation_model_0]

for x in range(1, 12):
        layer_outputs_x = [layer.output for layer in cifar_model.layers[x: (x + 1)]]
        activation_model_x = keras.models.Model(inputs=models[x-1].output, outputs=layer_outputs_x)
        models.append(activation_model_x)

# layer_outputs_2 = [layer.output for layer in cifar_model.layers[1:2]]
#
# activation_model_2 = keras.models.Model(inputs=activation_model_1.output, outputs=layer_outputs_2)
#
# layer_outputs_3 = [layer.output for layer in cifar_model.layers[2:3]]
#
# activation_model_3 = keras.models.Model(inputs=activation_model_2.output, outputs=layer_outputs_3)

num_classes = 2

print(len(names))

results = []

activations = activation_model_0.predict_generator(val_gen, steps=step)
results.append(activations)

for x in range(1, len(models)):
        activations = models[x].predict(results)


print("activations: " + str(len(activations)))

for x in range(len(activations)):
        activation = activations[x]
        print(str(len(activation)))
        plt.imshow(activation[0, :, :, 4], cmap='viridis')
        plt.show()

predicted_classes = cifar_model.predict_generator(val_gen, steps=step)

print("Printing predictions for horizon model: [true | false]")
for x in range(len(names)):
    print("File " + names[x] + " result: " + str(predicted_classes[x]))

exit(0)
