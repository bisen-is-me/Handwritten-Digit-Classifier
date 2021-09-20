import cv2 as cv
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras import datasets,layers,models
from mnist import MNIST
import idx2numpy
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint


#CNN--MODEL
model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

#dataset-importing

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-3\\mnistTask',
    target_size=(28,28),
    batch_size=14,
    shuffle=True,
    color_mode = "grayscale",
    class_mode='categorical'
  )

valid_generator=train_datagen.flow_from_directory(
    directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-3\\mnistTask',
    target_size=(28,28),
    batch_size=60,
    shuffle=True,
    color_mode = "grayscale",
    class_mode='categorical'
  )
#Testing data

mnist=tf.keras.datasets.mnist

(x_train_image, y_train_image),(test_images,test_labels)= datasets.mnist.load_data()

x_train_image,test_images = x_train_image/255.00 ,test_images/255.00

test_images= test_images.reshape((-1, 28, 28, 1))

test_labels=to_categorical(test_labels)


#saving the model checkpoints
filepath="C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-3\\t3_best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#ending work

model_fit = model.fit(train_generator,epochs = 150,validation_data=valid_generator,callbacks=callbacks_list)
#analysing with/without checkpoints

# accuracy on validation set
print("Testing:")
loss,accuracy=model.evaluate(test_images,test_labels)
print("Printing loss and accuracy before loading model weights")
print('Loss:',loss)
print('Accuracy:',100*(accuracy))

# loading the best model
model.load_weights("C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-3\\t3_best_weights.hdf5")

# accuracy on validation set
print("")
print("Testing:")
loss,accuracy=model.evaluate(test_images,test_labels)
print("Printing loss and accuracy after loading model weights")
print('Loss:',loss)
print('Accuracy:',100*(accuracy))
