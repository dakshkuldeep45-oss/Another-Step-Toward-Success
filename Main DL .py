# STEP 1: Import core Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# STEP 2: Load the standard Handwritten Digit Dataset (MNIST)
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalise pixel values from (0 to 255) to (0.0 to 1.0)
X_train, X_test = X_train / 255.0, X_test / 255.0

print("--- Dataset Loaded Successfully ---")
print(f"Training Data Shape: {X_train.shape}")
print("-----------------------------------")

# STEP 3: Build the Deep Learning Neural Network Brain
model = Sequential([
    Flatten(input_shape=(28, 28)),    # Input Layer: Flattens 2D grid into a single line
    Dense(32, activation='relu'),     # Hidden Layer: 32 nodes analyzing image lines
    Dense(10, activation='softmax')   # Output Layer: 10 nodes for digits 0-9
])

# STEP 4: Compile the Neural Network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# STEP 5: Train the Neural Network Brain
print("\n--- Training the Artificial Neural Network ---")
model.fit(X_train, y_train, epochs=3)

# STEP 6: Evaluate the Model on Test Data
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nFinal Model Accuracy on Test Images: {test_acc * 100:.2f}%")

# STEP 7: Practical Inference Testing
sample_image = X_test[0]
actual_label = y_test[0]

prediction = model.predict(sample_image.reshape(1, 28, 28), verbose=0)
predicted_label = tf.argmax(prediction[0]).numpy()

print(f"\n--- Production Inference Evaluation ---")
print(f"Actual Digit on Image: {actual_label}")
print(f"🧠 Neural Network Predicted Digit: {predicted_label}")
