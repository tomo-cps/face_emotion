import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import load_imgclasses = ({0:'angry',1:'disgust',2:'fear',3:'happy',
        4:'sad',5:'surprise',6:'neutral'})

image_path = '/Users/okawatomoaki/Documents/face_predict/images/happy.jpg'
#image_path = '.images/sad.jpg'

#img = image.load_img(image_path, grayscale=False , target_size=(64, 64))
print(dir(image.load_img))

'''

print('===========')
print(img)

img_array = image.img_to_array(img)

print(img_array.shape)

pImg = np.expand_dims(img_array, axis=0) / 255


model_path = '/Users/okawatomoaki/Documents/face_predict/trained_models/fer2013_mini_XCEPTION.110-0.65.hdf5'

emotions_XCEPTION = load_model(model_path, compile=False)

prediction = emotions_XCEPTION.predict(pImg)[0]

#convert the model into tf.js model
save_path = '../nodejs/static/emotion_XCEPTION'
#tfjs.converters.save_keras_model(emotions_XCEPTION, save_path)
#print("[INFO] saved tf.js emotion model to disk..")

top_indices = prediction.argsort()[-5:][::-1]
result = [(classes[i] , prediction[i]) for i in top_indices]
for x in result:
    print(x)
'''
