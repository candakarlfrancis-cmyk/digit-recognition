import numpy as np
from tensorflow import keras
from PIL import Image

# ✅ Load trained CNN model
model = keras.models.load_model("cnn_model.keras")

# ✅ Load image
img = Image.open("7.png").convert("L")  # grayscale
img = img.resize((28, 28))

# ✅ Convert to array
img = np.array(img)

# Invert colors (VERY IMPORTANT for MNIST)
img = 255 - img

# Normalize
img = img / 255.0

# Reshape for CNN
img = img.reshape(1, 28, 28, 1)

# ✅ Predict
prediction = model.predict(img)
digit = np.argmax(prediction)

print("Predicted Digit:", digit)