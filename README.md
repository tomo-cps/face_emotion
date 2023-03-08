# Facial Expression Estimation with OpenCV

## Introduction
This repository is a real-time facial expression estimation using OpenCV.

## Reference:
1. <https://github.com/PonDad/manatee/tree/master/2_emotion_recognition-master/python>

## Setting Up Environment
The Python environment is python==3.7.0
```
$ conda create -n face_predict python==3.7.0
$ conda activate face_predict
```
```
$ git clone https://github.com/tomo-cps/face_predict.git
```
```
$ pip install -r requirements.txt
```

Use this .xml file "haarcascade_frontalface_default.xml" for face detection in predict2.py

## Facial Expression Estimation with OpenCV
The following command would perform a real-time facial expression estimation based on the expression made.
```                                                   
$ python predict2.py
```
