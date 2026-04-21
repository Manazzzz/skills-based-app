# TensorFlow

## Level 1: Metadata

* **Skill Name:** TensorFlow
* **Category:** Deep Learning / Machine Learning
* **Level:** Intermediate–Advanced
* **Author:** Manas Malviya

---

## Level 2: Instructions

### 🔹 Installation

```bash
pip install tensorflow
```

---

### 🔹 Basic Workflow

1. Load dataset
2. Preprocess (normalize, reshape, tokenize)
3. Build model
4. Compile model
5. Train model
6. Evaluate & optimize

---

### 🔹 Artificial Neural Network (ANN)

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```

---

### 🔹 Convolutional Neural Network (CNN)

```python
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])
```

---

### 🔹 Recurrent Neural Network (RNN)

```python
model = keras.Sequential([
    keras.layers.SimpleRNN(64, return_sequences=False),
    keras.layers.Dense(1)
])
```

---

### 🔹 LSTM (Long Short-Term Memory)

```python
model = keras.Sequential([
    keras.layers.LSTM(64, return_sequences=False),
    keras.layers.Dense(1)
])
```

---

### 🔹 GRU (Gated Recurrent Unit)

```python
model = keras.Sequential([
    keras.layers.GRU(64),
    keras.layers.Dense(1)
])
```

---

### 🔹 Variational Autoencoder (VAE - Concept)

```python
# Encoder
encoder = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(2)  # latent space
])

# Decoder
decoder = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(784, activation='sigmoid')
])
```

---

### 🔹 Generative Adversarial Network (GAN - Concept)

```python
# Generator
generator = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(784, activation='tanh')
])

# Discriminator
discriminator = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

---

### 🔹 Model Training

```python
model.fit(x_train, y_train, epochs=5, validation_split=0.2)
```

---

### 🔹 Evaluation

```python
loss, acc = model.evaluate(x_test, y_test)
print(acc)
```

---

### 🔹 Model Saving & Loading

```python
model.save("model.h5")
loaded = keras.models.load_model("model.h5")
```

---

### 🔹 TensorFlow Lite (Deployment)

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

---

### 🔹 Debugging

```python
tf.debugging.check_numerics(tensor, "NaN detected")
```

---

## Level 3: Resources & Code

### 📚 Documentation

* https://www.tensorflow.org/
* https://keras.io/
* skills.sh

---

### 📊 Use Cases

* Image Classification (CNN)
* NLP & Text (RNN, LSTM, GRU)
* Time Series Forecasting
* Anomaly Detection
* Generative Models (VAE, GANs)

---

### ⚠️ Best Practices

* Normalize input data
* Use Dropout (0.2–0.5)
* Monitor validation loss
* Tune learning rate
* Use EarlyStopping
* Start simple, then scale

---

### ❗ Common Mistakes

* Not normalizing data
* Wrong loss function
* Overfitting small datasets
* Shape mismatch errors
* High learning rate → NaN loss

---
