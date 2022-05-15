import cv2
import tensorflow as tf
import numpy as np 
import csv 




tflite_model_path = "/home/chrisus/Tensorflow/Emociones/modelo/model.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']

fields = ['Enojo','Disgusto','Miedo'
        ,'Feliz'
        ,'Neutral'
        ,'Triste'
        ,'Sorprendido'] 
filename = "records.csv"

row  = [0,0,0
        ,0
        ,0
        ,0
        ,0] 


def Graph(fields_g, row_g):
	x_pos = np.arange(len(fields_g))
	# Parametros de las barras
	plt.bar(x_pos, row_g, color=['red', 'green', 'purple', 'yellow', 'gray', 'blue', 'pink'])
	# Eje X
	plt.xticks(x_pos, fields_g)
	# Grafica
	plt.xlabel("Emociones")
	plt.ylabel("Cantidad")
	plt.title("Cantidad de emociones detectadas")
	#plt.show()

	# dd/mm/YY H:M:S
	now = datetime.now()
	dt_string = now.strftime("%d%m%Y_%H%M%S")
	name = '{0}{1}{2}'.format("EmotDet_", dt_string, ".png")
	plt.savefig(name)


# Load image
cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
aux=emotion_dict[4] 
    # start the webcam feed
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        input_data = cropped_img 
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        output_data_tflite = interpreter.get_tensor(output_details[0]['index'])

        maxindex= int(np.argmax(output_data_tflite,axis=1))
        
        if aux!=emotion_dict[maxindex]:
            row[maxindex]+=1
        aux=emotion_dict[maxindex]    
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

with open(filename, 'w') as csvfile: 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerow(row)
cap.release()
Graph(fields, row)
cv2.destroyAllWindows()
