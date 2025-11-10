import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the saved model
model = load_model(r"D:\cnn_prj\outputs\my_model.h5")

# Path to your test image (put your test image in D:\cnn_prj\test folder)
img_path = r"D:\cnn_prj\test\dog3.jpeg"

# Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
label = "Dog" if prediction[0] > 0.5 else "Cat"

print(f"Prediction: {label}")
plt.imshow(img)
plt.title(f"Prediction: {label}")
plt.axis('off')
plt.show()
