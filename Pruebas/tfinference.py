import numpy as np
import tensorflow as tf
import cv2
from keras.preprocessing.image import load_img, img_to_array
from time import time
from matplotlib import pyplot as plt
import os
from keras.utils.np_utils import normalize


#Parasitized image
#image = load_img('cell_images/Parasitized/C39P4thinF_original_IMG_20150622_111206_cell_99.png', target_size=(150,150))

#Uninfected image
image = cv2.imread('/home/chrisus/Tensorflow/emotion-detection/Emotion-detection/src/im0.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # ADD THIS
img = cv2.resize(gray,(48, 48))# ADD THIS
image = img_to_array(img)
# reshape data for the model
image = image.reshape((1, 48,48, 1))
print(image.shape)


image = normalize(image, axis=0)
print(image.shape)

###############################################
###PREDICT USING REGULAR KERAS TRAINED MODEL FILE (h5). 
##########################################################

keras_model_size = os.path.getsize("/home/chrisus/em.h5")/1048576  #Convert to MB
print("Keras Model size is: ", keras_model_size, "MB")
#Using regular keral model
model =tf.keras.models.load_model("/home/chrisus/em.h5")

time_before=time()
keras_prediction = np.argmax(model.predict(image),axis=1)
time_after=time()
total_keras_time = time_after - time_before
print("Total prediction time for keras model is: ", total_keras_time)

print("The keras prediction for this image is: ", keras_prediction, " 0=Uninfected, 1=Parasited")




##################################################################################
#### PREDICT USING tflite ###
############################################################################
tflite_size = os.path.getsize("/home/chrisus/Tensorflow/emotion-detection/Emotion-detection/src/model.tflite")/1048576  #Convert to MB
print("tflite Model without opt. size is: ", tflite_size, "MB")
#Not optimized (file size = 540MB). Taking about 0.5 seconds for inference
tflite_model_path = "/home/chrisus/Tensorflow/emotion-detection/Emotion-detection/src/model.tflite"

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Load image
input_data = image
interpreter.set_tensor(input_details[0]['index'], input_data)

time_before=time()
interpreter.invoke()
time_after=time()
total_tflite_time = time_after - time_before
print("Total prediction time for tflite without opt model is: ", total_tflite_time)

output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
output_data_tflite= np.argmax(output_data_tflite,axis=1)
print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")

#################################################################
#### PREDICT USING tflite with optimization###
#################################################################
tflite_optimized_size = os.path.getsize("/home/chrisus/Tensorflow/emotion-detection/Emotion-detection/src/em.tflite")/1048576  #Convert to MB
print("tflite Model with optimization size is: ", tflite_optimized_size, "MB")
#Optimized using default optimization strategy (file size = 135MB). Taking about 39 seconds for prediction
tflite_optimized_model_path = "/home/chrisus/Tensorflow/emotion-detection/Emotion-detection/src/em.tflite"



# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_optimized_model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Load image
input_data = image

interpreter.set_tensor(input_details[0]['index'], input_data)

time_before=time()
interpreter.invoke()
time_after=time()
total_tflite_opt_time = time_after - time_before
print("Total prediction time for tflite model with opt is: ", total_tflite_opt_time)

output_data_tflite_opt = interpreter.get_tensor(output_details[0]['index'])
output_data_tflite_opt= np.argmax(output_data_tflite_opt,axis=1)
print("The tflite with opt prediction for this image is: ", output_data_tflite_opt, " 0=Uninfected, 1=Parasited")

#############################################

#Summary
print("###############################################")
print("Keras Model size is: ", keras_model_size)
print("tflite Model without opt. size is: ", tflite_size)
print("tflite Model with optimization size is: ", tflite_optimized_size)
print("________________________________________________")
print("Total prediction time for keras model is: ", total_keras_time)
print("Total prediction time for tflite without opt model is: ", total_tflite_time)
print("Total prediction time for tflite model with opt is: ", total_tflite_opt_time)
print("________________________________________________")
print("The keras prediction for this image is: ", keras_prediction, " 0=Uninfected, 1=Parasited")
print("The tflite w/o opt prediction for this image is: ", output_data_tflite, " 0=Uninfected, 1=Parasited")
print("The tflite with opt prediction for this image is: ", output_data_tflite_opt, " 0=Uninfected, 1=Parasited")