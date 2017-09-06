from keras.models import load_model
from keras.utils import plot_model
from keras import backend as K
import cv2
import numpy as np
import gc
import argparse

gc.collect()
K.clear_session()

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--directory", required=True,
	help="path to the input image")
args = vars(ap.parse_args())

model = load_model('model/model.h5')
#
plot_model(model, to_file='model/model.png')

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

countImage = 0
predictTrue = 0
for file in os.listdir(args["directory"]):
    #   print (file)
    img = cv2.imread(args["directory"] + "/" + file)
    img = cv2.resize(img,(500,500))
    img = np.reshape(img,[1,500,500,3])

    classes = model.predict_classes(img)
    type = file.split("_")[-2]
    predicted = ""
    countImage += 1
    print (classes)
    if (classes[0] == 0):
        print (file + ": Glass")
        predicted = "Glass"
    elif (classes[0] == 1):
        print(file + ": Bottle")
        predicted = "Bottle"
    else:
        print(file + ": Spark Plug")
        predicted = "Spark_Plug"
    #print (type)
    #print (predicted)
    if(type == predicted):
        predictTrue += 1

print ("")
print ("")
print ("")
print ("Total Images: " + str(countImage))
print ("Correct Prediction: " + str(predictTrue))
print ("Model Accuracy: " + str((predictTrue / countImage)*100))
