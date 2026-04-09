import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras

# ================================
# ✅ LOAD MODEL
# ================================
model = keras.models.load_model("cnn_model.keras")

# ================================
# ✅ CREATE WINDOW
# ================================
window = tk.Tk()
window.title("Digit Recognition System")
window.geometry("400x500")

# ================================
# ✅ CANVAS (DRAW AREA)
# ================================
canvas = tk.Canvas(window, width=280, height=280, bg='white')
canvas.pack(pady=20)

# Image for drawing
image = Image.new("L", (280, 280), color=255)
draw = ImageDraw.Draw(image)

# ================================
# ✏️ DRAW FUNCTION
# ================================
def draw_lines(event):
    x, y = event.x, event.y
    r = 8  # brush size
    canvas.create_oval(x-r, y-r, x+r, y+r, fill='black')
    draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

canvas.bind("<B1-Motion>", draw_lines)

# ================================
# 🧠 PREDICT FUNCTION
# ================================
def predict_digit():
    # Resize to 28x28
    img = image.resize((28, 28))

    # Convert to array
    img = np.array(img)

    # Invert colors (important)
    img = 255 - img

    # Normalize
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)
    digit = np.argmax(prediction)

    result_label.config(text=f"Prediction: {digit}")

# ================================
# 🧹 CLEAR CANVAS
# ================================
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, 280, 280], fill=255)
    result_label.config(text="Prediction: ")

# ================================
# 📂 UPLOAD IMAGE
# ================================
def upload_image():
    file_path = filedialog.askopenfilename()

    if file_path:
        img = Image.open(file_path).convert("L")
        img = img.resize((28, 28))

        img_array = np.array(img)
        img_array = 255 - img_array
        img_array = img_array / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        result_label.config(text=f"Prediction: {digit}")

# ================================
# 🔘 BUTTONS
# ================================
btn_predict = tk.Button(window, text="Predict", command=predict_digit)
btn_predict.pack(pady=5)

btn_clear = tk.Button(window, text="Clear", command=clear_canvas)
btn_clear.pack(pady=5)

btn_upload = tk.Button(window, text="Upload Image", command=upload_image)
btn_upload.pack(pady=5)

# ================================
# 📊 RESULT LABEL
# ================================
result_label = tk.Label(window, text="Prediction: ", font=("Arial", 16))
result_label.pack(pady=20)

# ================================
# ▶️ RUN APP
# ================================
window.mainloop()