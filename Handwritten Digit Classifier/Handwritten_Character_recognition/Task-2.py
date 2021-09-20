import tensorflow as tf
from tensorflow.keras import datasets,layers,models
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

mnist=tf.keras.datasets.mnist
(x_train_image, y_train_image),(x_test_image, y_test_image)= datasets.mnist.load_data()
x_train_image,x_test_image = x_train_image/255.00 , x_test_image/255.00

#PREPROCESSING
x_train_image = x_train_image.reshape((60000, 28, 28, 1))
x_test_image= x_test_image.reshape((10000, 28, 28, 1))

#1-h encode
y_train_image = to_categorical(y_train_image)
y_test_image = to_categorical(y_test_image)

#CNN--MODEL
model = models.Sequential()
model.add(layers.Conv2D(4, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Conv2D(4, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=2))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

#saving the model checkpoints
filepath="C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-2\\t2_best_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#fitting work
'''
model_fit = model.fit(x_train_image, y_train_image, epochs=40,batch_size=32,shuffle=True, validation_data=(x_test_image,y_test_image),callbacks=callbacks_list)
'''
#analysing with/without checkpoints

# accuracy on validation set
print("Testing:")
loss,accuracy=model.evaluate(x_test_image,y_test_image)
print("Printing loss and accuracy before loading model weights")
print('Loss:',loss)
print('Accuracy:',100*(accuracy))

# loading the best model
model.load_weights("C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-2\\t2_best_weights.hdf5")

# accuracy on validation set
print("")
print("Testing with the saved checkpoints")
loss,accuracy=model.evaluate(x_test_image,y_test_image)
print("Printing loss and accuracy after loading model weights")

print('Loss:',loss)
print('Accuracy:',100*(accuracy))
'''
#TASK---1
#couldn't perform loading dataset form task-1 and then evaluating
print("Loading model from Task-1")
model.load_weights("C:\\Users\\ASUS\\Documents\\UAS-SOFTWARE-TASK-2\\Handwritten Digit Classifier\\Handwritten_Character_recognition\\Model Checkpoints\\Task-1\\t1_best_weights.hdf5")

# accuracy on validation set
print("")
print("Testing with the saved checkpoints")
loss,accuracy=model.evaluate(x_test_image,y_test_image)
print("Printing loss and accuracy after loading model weights")

print('Loss:',loss)
print('Accuracy:',100*(accuracy))
'''
