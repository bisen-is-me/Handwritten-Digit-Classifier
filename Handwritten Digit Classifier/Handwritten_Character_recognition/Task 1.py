import cv2 as cv
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt
import os
from tensorflow.keras import datasets,layers,models
import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score



#CNN--MODEL
model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(36, activation='softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()
#dataset-importing

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)


train_generator=train_datagen.flow_from_directory(
    directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\train',
    target_size=(28,28),
    batch_size=1,
    shuffle=True,
    color_mode = "grayscale",
    class_mode='categorical'
  )
valid_generator=train_datagen.flow_from_directory(
    directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\train',
    target_size=(28,28),
    batch_size=32,
    shuffle=True,
    color_mode = "grayscale",
    class_mode='categorical'
  )
test_generator= test_datagen.flow_from_directory(
    directory='C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Dataset-Task\\Task-1\\test',
    target_size=(28,28),
    batch_size= 1,
    shuffle=True,
    color_mode="grayscale",
    class_mode='categorical'

)
#saving the model checkpoints
filepath="C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-1\\t1_best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#ending work
model_fit = model.fit(train_generator,epochs = 40,shuffle=True,validation_data=valid_generator,callbacks=callbacks_list)


#analysing with/without checkpoints

# accuracy on validation set
print("Testing:")
loss,accuracy=model.evaluate(test_generator)
print("Printing loss and accuracy before loading model weights")
print('Loss:',loss)
print('Accuracy:',100*(accuracy))

# loading the best model
model.load_weights("C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-1\\t1_best_weights.hdf5")

# accuracy on validation set
print("")
print("Testing:")
loss,accuracy=model.evaluate(test_generator)
print("Printing loss and accuracy after loading model weights")
print('Loss:',loss)
print('Accuracy:',100*(accuracy))
'''
#testing result--tried to add a functionality

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

'''
