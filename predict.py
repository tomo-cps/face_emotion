from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import cv2  

#def facial expression
classes = ({0:'angry',
            1:'disgust',
            2:'fear',
            3:'happy',
            4:'sad',
            5:'surprise',
            6:'neutral'})

#train face
face_cascade_file = './haarcascade_frontalface_default.xml'  
front_face_detector = cv2.CascadeClassifier(face_cascade_file)  

#camera
cap = cv2.VideoCapture(0)

#train facial expression
model_path = './trained_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
emotions_XCEPTION = load_model(model_path, compile=False)

minW = 0.1*cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
minH = 0.1*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  

while True:
    tick = cv2.getTickCount()  
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = front_face_detector.detectMultiScale(   
        gray,  
        scaleFactor = 1.2,  
        minNeighbors = 3,  
        minSize = (int(minW), int(minH)),  
       )

    #Multiple faces can be detected
    for(x,y,w,h) in faces:   
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  
        face_cut = img[y:y+h, x:x+w]
        face_cut = cv2.resize(face_cut,(64, 64))
        face_cut = cv2.cvtColor(face_cut, cv2.COLOR_BGR2GRAY)
        img_array = image.img_to_array(face_cut)
        pImg = np.expand_dims(img_array, axis=0) / 255
        
        #face prediction
        prediction = emotions_XCEPTION.predict(pImg)[0]
        save_path = '../nodejs/static/emotion_XCEPTION'

        top_indices = prediction.argsort()[-5:][::-1]

        result = sorted([[classes[i], prediction[i]] for i in top_indices], reverse=True, key=lambda x: x[1])
        result_c = result[0][0]

        cv2.putText(img, str(result_c), (x+5,y-5), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        
    # FPS算出と表示用テキスト作成  
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)  
    # FPS  
    cv2.putText(img, "FPS:{} ".format(int(fps)),   
        (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)  
      
    cv2.imshow('camera',img)
    # ESC
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n Exit Program")
cap.release()
cv2.destroyAllWindows()
