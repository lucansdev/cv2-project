from PIL import Image
import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
caminhos = [os.path.join("pictures_new_ai",f) for f in os.listdir("pictures_new_ai")]

target_size = (64,64)

ids = [2,2,2,2,3,3,3,3,3,4,4,4,4,4,4,4,4]
data_file = []
for new_images in caminhos:
    with Image.open(new_images) as image:
        image = image.convert("L")
        image_resized = image.resize(target_size)
        image_np = np.array(image_resized)
        imagem_np = [value for i in image_np for value in i]
        data_file.append(imagem_np)


data_file = np.array(data_file)

stander = StandardScaler()
data_file = stander.fit_transform(data_file)


ai = MLPClassifier(max_iter=1000,hidden_layer_sizes=(10,10,10,10),activation="relu",solver="adam")
ai.fit(data_file,ids)


with open("AI_better.sav","wb") as file:
    pickle.dump(ai,file)
