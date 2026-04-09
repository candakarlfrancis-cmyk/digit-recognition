import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input

# ================================
# ✅ LOAD DATASET
# ================================
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
print("Dataset loaded!")

# ================================
# ✅ PREPROCESSING
# ================================

# Normalize (0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# Flatten for ANN
x_train_ann = x_train.reshape(-1, 784)
x_test_ann = x_test.reshape(-1, 784)

# Reshape for CNN
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

# One-hot encode labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Preprocessing complete!")

# ================================
# ✅ BUILD ANN MODEL
# ================================
ann_model = Sequential([
    Input(shape=(784,)),  # FIXED (no warning)
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile ANN
ann_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train ANN
print("Training ANN...")
ann_model.fit(
    x_train_ann, y_train,
    epochs=5,
    validation_data=(x_test_ann, y_test)
)

# Save ANN model
ann_model.save("ann_model.keras")
print("ANN Training Complete!")

# ================================
# ✅ BUILD CNN MODEL
# ================================
cnn_model = Sequential([
    Input(shape=(28, 28, 1)),  # FIXED (no warning)
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile CNN
cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train CNN
print("Training CNN...")
cnn_model.fit(
    x_train_cnn, y_train,
    epochs=5,
    validation_data=(x_test_cnn, y_test)
)

# Save CNN model
cnn_model.save("cnn_model.keras")
print("CNN Training Complete!")