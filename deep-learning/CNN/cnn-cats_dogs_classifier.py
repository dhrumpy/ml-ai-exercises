#!/usr/bin/env python
# coding: utf-8

# importing keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

classifier = Sequential()

classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

# refer to lec 62 for these additional layers
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

classifier.add(Flatten())

classifier.add(Dense(units=128, activation='relu')) 
classifier.add(Dense(units=1, activation='sigmoid')) 
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

import time
start = time.time()
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_path = '/home/dhrumpy/Downloads/original/DL Colab Changes/Convolutional_Neural_Networks 3/dataset/training_set'
# class_mode: classification categories
training_set = train_datagen.flow_from_directory(training_path,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_path = '/home/dhrumpy/Downloads/original/DL Colab Changes/Convolutional_Neural_Networks 3/dataset/test_set'
test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=(64,64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000,
                         epochs=1,
                         validation_data=test_set,
                         validation_steps=2000)
end = time.time()
print('Finished in {} seconds.'.format(end-start))