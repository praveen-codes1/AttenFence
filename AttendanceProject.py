import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path= 'ImagesAttendance'
images= []
classNames= []
myList = os.listdir(path)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('attendance.csv', 'r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now=datetime.now()
            dtString=now.strftime("%H:%M:%S")
            f.writelines(f'\n{name},{dtString}')




encodeListKnown=findEncodings(images)
print('Encoding complete!')

cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrFrame = face_recognition.face_locations(imgS)
    encodesCurrFrame = face_recognition.face_encodings(imgS,facesCurrFrame)

    for encodeFace,faceLoc in zip(encodesCurrFrame,facesCurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)



        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1=faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img, (x1, y1), (x2, y2-35), (0, 255, 0), 2)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)
            markAttendance(name)


    cv2.imshow('Webcam', img)
    cv2.waitKey(1)


#from basics
#faceloc=face_recognition.face_locations(imgAr)[0]
#encodeAr=face_recognition.face_encodings(imgAr)[0]
#cv2.rectangle(imgAr,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

#faceloct=face_recognition.face_locations(imgT)[0]
#encodeArt=face_recognition.face_encodings(imgT)[0]
#cv2.rectangle(imgT,(faceloct[3],faceloct[0]),(faceloct[1],faceloct[2]),(255,0,255),2)

#results=face_recognition.compare_faces([encodeAr],encodeArt)
#facedis=face_recognition.face_distance([encodeAr],encodeArt)
#print(facedis)
#cv2.putText(imgT,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)