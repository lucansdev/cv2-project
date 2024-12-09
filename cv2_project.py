import cv2
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
rastreador = cv2.TrackerCSRT_create()
with open("AI_better.sav","rb") as file:
    ai:MLPClassifier = pickle.load(file)
    

video = cv2.VideoCapture(0)
ok,frames = video.read()

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

imagem = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

deteccoes  = detector.detectMultiScale(imagem,scaleFactor=1.1,minSize=(30,30))

okk = rastreador.init(frames,*deteccoes)

for (x,y,w,h) in deteccoes:
    little_face = imagem[y:y+h,x:x+w]

imagem_modificada = [value for i in little_face for value in i]
new_imagem = [imagem_modificada]

new_image = np.resize(new_imagem,4096).reshape(1,-1)

stander = StandardScaler()
new_image_standard = stander.fit_transform(new_image)

idprevisto = ai.predict(new_image_standard)

while True:
    oks,frame = video.read()
    if not oks:
        break

    okk,bbox = rastreador.update(frame)
    
    
    if okk:
        (x,y,w,h) = [int(v) for v in bbox]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2,1)
        if idprevisto[0] == 4:
            cv2.putText(frame,"lucas",(x,y+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)
        elif idprevisto[0] == 2:
            cv2.putText(frame,"bruna",(x,y+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)
        elif idprevisto[0] == 3:
            cv2.putText(frame,"mamae",(x,y+30),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)

    else:
        cv2.putText(frame,"falha no rastreamento",(100,80),cv2.FONT_HERSHEY_COMPLEX_SMALL,0.75,(0,0,255),2)
        exit()

    cv2.imshow("rastreando",frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
