from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import gc

gc.collect()

model = Sequential()
model.add(Convolution2D(128, (3, 3), input_shape=(500, 500, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Convolution2D(64, (3, 3), dim_ordering="th"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Convolution2D(64, (3, 3), dim_ordering="th"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),dim_ordering="th"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('softmax'))

# model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(128, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(256, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(ZeroPadding2D((1, 1)))
# model.add(Convolution2D(512, 3, 3, activation='relu'))
# model.add(MaxPooling2D((2, 2), strides=(2, 2)))
#
# model.add(Flatten())
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(3, activation='softmax'))


# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


batch_size = 32

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'train_labeling',  # this is the target directory
        target_size=(500, 500),  # all images will be resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels, categorical
print (train_generator.batch_size)

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test_labeling',
        target_size=(500, 500),
        batch_size=batch_size,
        class_mode='binary')

print (validation_generator.class_indices)
print (validation_generator.classes)
model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('first_try.h5')  # always save your weights after training or during training

model.save('model.h5')  # always save your weights after training or during training


