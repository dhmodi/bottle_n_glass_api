from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
import argparse
import cv2
import numpy as np


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--directory", required=True,
	help="path to the input image")
args = vars(ap.parse_args())


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
arg1 = args["directory"]
for file in os.listdir(arg1):
    print (file)
    img = cv2.imread(arg1 + "/" + file)  # this is a PIL image
    # x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir='partRecognition/train_labeling/'+arg1+'/', save_prefix=arg1, save_format='jpeg'):
        i += 1
        if i > 20:
            break  # otherwise the generator would loop indefinitely