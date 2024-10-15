import cv2 
from PIL import Image
import os
import numpy as np
def dados_imagem():
    caminhos = [os.path.join("fotos_minha",f) for f in os.listdir("fotos_minha")]
    faces = []
    ids = []
    for caminho in caminhos:
        imagem = Image.open(caminho).convert("L")
        imagem_np = np.array(imagem,"uint8")
        id = int(os.path.split(caminho)[1].split('.')[0].replace('subject', ''))
        ids.append(id)
        faces.append(imagem_np)
    return np.array(ids),faces

ids,faces = dados_imagem()


lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces,ids)
lbph.write("classificador_eu.yml")

reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificador_eu.yml")

imagem_teste = f"{os.path.join("fotos_minha\subject03.nada.gif")}"

imagem = Image.open(imagem_teste).convert("L")
imagem_np = np.array(imagem,"uint8")

idprevisto,_ = reconhecedor.predict(imagem_np)
print(idprevisto)