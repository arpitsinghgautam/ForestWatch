from tensorflow.keras.models import load_model
import os
import random
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the entire model
loaded_model = load_model('deforestation_model.h5')

img_path = "amazon-satellite-images/test-jpg-additional/test-jpg-additional/file_100.jpg"
img_size=(224, 224)
img = image.load_img(img_path, target_size=img_size)
x = image.img_to_array(img)
print("LENGTH OF THE IMAGE ARRAY IS: ",len(x))
x = np.expand_dims(x, axis=0)
x = x / 255.0
print("LENGTH OF THE IMAGE ARRAY after: ",len(x))
preds = loaded_model.predict(x)
print(preds)
deforestation_prob = preds[0][0]
pollution_prob = preds[0][1]
img = image.load_img(img_path)

plt.figure()
plt.imshow(img)
plt.title(f"Deforestation Probability: {deforestation_prob:.4f}\nPollution Probability: {pollution_prob:.4f}")
plt.axis("off")
plt.show()