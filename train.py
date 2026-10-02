import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from datasets import load_dataset
import pickle
import os

# 1. Load the HuggingFace dataset
print("Loading dataset dair-ai/emotion...")
dataset = load_dataset("dair-ai/emotion")

train_data = dataset['train']

X_list = list(train_data['text'])
X = tf.convert_to_tensor(X_list, dtype=tf.string)
y = np.array(train_data['label'])

# Labels for dair-ai/emotion: 0: sadness, 1: joy, 2: love, 3: anger, 4: fear, 5: surprise
index_to_label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}
labels = list(index_to_label.values())

# 2. Text Vectorization
max_tokens = 10000
max_len = 100

vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=max_len)
vectorizer.adapt(X)

# 3. Build Model
model = Sequential([
    vectorizer,
    Embedding(input_dim=max_tokens, output_dim=16),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(len(labels), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 4. Train Model
print("Training model... this might take a minute.")
model.fit(X, y, epochs=10, validation_split=0.1)
print("Training completed.")

# 5. Save Model and Metadata
model_dir = "saved_model"
os.makedirs(model_dir, exist_ok=True)
model.save(os.path.join(model_dir, "emotion_model.keras"))

with open(os.path.join(model_dir, "label_mapping.pkl"), "wb") as f:
    pickle.dump(index_to_label, f)

print(f"Model saved to {model_dir}")
