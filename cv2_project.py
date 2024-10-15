import cv2
from PIL import Image
import numpy as np
rastreador = cv2.TrackerCSRT_create()
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classifier_me.yml")

video = cv2.VideoCapture(0)
ok,frame = video.read()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imagem = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

deteccoes  = detector.detectMultiScale(imagem,1.2)

ok = rastreador.init(frame,*deteccoes)


idprevisto,_ = reconhecedor.predict(imagem)
print(idprevisto)

while True:
    ok,frame = video.read()
    if not ok:
        break

    ok,bbox = rastreador.update(frame)
    
    
    if ok:
        (x,y,w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
        if idprevisto == 1:
            cv2.putText(frame,"lucas",(x,y+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)
        elif idprevisto == 2:
            cv2.putText(frame,"bruna",(x,y+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)
        elif idprevisto == 3:
            cv2.putText(frame,"mamae",(x,y+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)

    else:
        cv2.putText(frame,"falha no rastreamento",(100,80),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)

    cv2.imshow("rastreando",frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

