import cv2

image = cv2.imread("/usr/bin/beauty.jpg")
print('Se carga la imagen') 
eye_cascade = cv2.CascadeClassifier('/usr/bin/haarcascade_eye.xml')
print('Se carga xml') 
eyes = eye_cascade.detectMultiScale(image, scaleFactor = 1.2,
                                    minNeighbors = 4 )
 
 
for (x,y,w,h) in eyes:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0, 255, 0),5)
 
cv2.imwrite("Eye_Detected.jpg", image)
print('Se ejecuta eyedetect.py')
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Fin')
