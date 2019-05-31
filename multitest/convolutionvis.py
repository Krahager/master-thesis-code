import keras
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

#xrange = range
#izip = zip
#imap = map


directory = "/home/krahager/PyUiTestResults/MultiDetectionTest/demo"
model_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'wells_full_vgg.hdf5'

model_path = os.path.join(model_dir, model_name)

shape=(224,224)
resize_shape=(shape[0],shape[1],3)

#Init Imagedatagenerator

datagen = ImageDataGenerator(validation_split=None)

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

layer_outputs_1 = [layer.output for layer in cifar_model.layers[:32]]

activation_model_1 = keras.models.Model(inputs=cifar_model.input, outputs=layer_outputs_1)

num_classes = 2

print(len(names))

activations = activation_model_1.predict_generator(val_gen, steps=step)

print("activations: " + str(len(activations)))

path = '/home/krahager/PycharmProjects/Filters/horizon_plane_filter_'

for x in range(len(activations)):
        activation = activations[x]
        print(str(len(activation)))

        print(activation.shape)
        for y in range(activation.shape[3]):
                plt.imsave((path + str(x) + '_f' + str(y) + '.png'), activation[0, :, :, y], cmap='viridis')
                plt.show()

predicted_classes = cifar_model.predict_generator(val_gen, steps=step)

print("Printing predictions for horizon model: [true | false]")
for x in range(len(names)):
    print("File " + names[x] + " result: " + str(predicted_classes[x]))

exit(0)
