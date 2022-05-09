import tensorflow as tf

model=tf.keras.models.load_model('/home/chrisus/em.h5')

converter=tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations=[tf.lite.Optimize.DEFAULT]
tflite_model=converter.convert()
open('em.tflite', 'wb').write(tflite_model)