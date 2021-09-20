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
from tensorflow.keras import regularizers

#CNN--MODEL
model = models.Sequential()
#layer 1
model.add(layers.Conv2D(filters=32,kernel_size=5, strides=1, activation='relu', input_shape=(32, 32, 1),kernel_regularizer=tf.keras.regularizers.l2(0.0005)))

#layer 2
model.add(layers.Conv2D(filters=32,kernel_size=5, strides=1, use_bias=False))
#layer 3
model.add(layers.BatchNormalization())

#-----
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=2))
model.add(layers.Dropout(0.25))
#-----
#layer 3
model.add(layers.Conv2D(filters=64,kernel_size=3,strides=1,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.0005)))

#layer 4
model.add(layers.Conv2D(filters=64,kernel_size=3, strides=1, use_bias=False))
#layer 5
model.add(layers.BatchNormalization())
#-----
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=2))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
#-----
#layer 6
model.add(layers.Dense(units=256,use_bias=False))

#layer 7
model.add(layers.BatchNormalization())
#layer 8
model.add(layers.Dense(units=128,use_bias=False))
#layer 9
model.add(layers.BatchNormalization())
#layer 10
model.add(layers.Activation('relu'))
#layer 11
model.add(layers.BatchNormalization())
#Layer 12
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.25))
#output
model.add(layers.Dense(units=10,activation='softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#dataset-importing

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
    directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-3\\mnistTask',
    target_size=(32,32),
    batch_size=28,
    color_mode = "grayscale",
    class_mode='categorical'
  )
mndata='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-3\\test\\t10k-images.idx3-ubyte'
test_arr=idx2numpy.convert_from_file(mndata)
test_arr=test_arr/255.00
test_arr=to_categorical(test_arr)
model_fit = model.fit(train_generator,epochs = 40)
loss,accuracy=model.evaluate(test_arr)
print((accuracy)*100)
print(loss)



#saving the model

model.save('Task-3')














'''
import numpy
import cv2
from keras.preprocessing.image import ImageDataGenerator


test_datagen = ImageDataGenerator(1./255)
test_generator= test_datagen.flow_from_directory(
directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\test\\test1')
img=cv2.imread(test_generator.directory.jpg)
cv2.imshow('image', img)
cv2.waitKey(0)

import cv2
import os
directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\test\\test1'

def load_images_from_folder(path='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\test\\test1'
):
    images = []
    for filename in os.listdir('C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\test\\test1'
):
        img = cv2.imread(os.path.join('C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\test\\test1'
,filename))
        if img is not None:
            images.append(img)
    return images
cv2.imshow('image',load_images_from_folder)
cv2.waitKey(0)
#testing result

def get_result(result):
    if result[0][0] == 1:
        return('0')
    elif result[0][1] == 1:
        return ('1')
    elif result[0][2] == 1:
        return ('2')
    elif result[0][3] == 1:
        return ('3')
    elif result[0][4] == 1:
        return ('4')
    elif result[0][5] == 1:
        return ('5')
    elif result[0][6] == 1:
        return ('6')
    elif result[0][7] == 1:
        return ('7')
    elif result[0][8] == 1:
        return ('8')
    elif result[0][9] == 1:
        return ('9')
    elif result[0][10] == 1:
        return ('A')
    elif result[0][11] == 1:
        return ('B')
    elif result[0][12] == 1:
        return ('C')
    elif result[0][13] == 1:
        return ('D')
    elif result[0][14] == 1:
        return ('E')
    elif result[0][15] == 1:
        return ('F')
    elif result[0][16] == 1:
        return ('G')
    elif result[0][17] == 1:
        return ('H')
    elif result[0][18] == 1:
        return ('I')
    elif result[0][19] == 1:
        return ('J')
    elif result[0][20] == 1:
        return ('K')
    elif result[0][21] == 1:
        return ('L')
    elif result[0][22] == 1:
        return ('M')
    elif result[0][23] == 1:
        return ('N')
    elif result[0][24] == 1:
        return ('O')
    elif result[0][25] == 1:
        return ('P')
    elif result[0][26] == 1:
        return ('Q')
    elif result[0][27] == 1:
        return ('R')
    elif result[0][28] == 1:
        return ('S')
    elif result[0][29] == 1:
        return ('T')
    elif result[0][30] == 1:
        return ('U')
    elif result[0][31] == 1:
        return ('V')
    elif result[0][32] == 1:
        return ('W')
    elif result[0][33] == 1:
        return ('X')
    elif result[0][34] == 1:
        return ('Y')
    elif result[0][35] == 1:
        return ('Z')

img=cv.imread('C:\\Users\\ASUS\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\T.png')
width,height=28,28
imgrez=cv.resize(img,(width,height))
cv.imshow('image',img)
cv.waitKey(0)
imgfin = imgrez.reshape(-1,28, 28, 1)
imgfin=np.array(imgfin)
predict=model.predict(imgfin)
print(np.argmax(predict))

result=get_result(np.argmax(predict[0][0]))
print(predict)
print("Probably the given image is:")
print(format(result))

'''
