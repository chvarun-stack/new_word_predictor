# train.py
# ----------------------------------------
# Next Word Prediction - Training Script
# ----------------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
import numpy as np
import os

# Load and preprocess data
file = open("data/Pride_and_Prejudice.txt", "r", encoding="utf8")
lines = file.readlines()

data = " ".join(lines)
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '')
data = data.replace('“', '').replace('”', '')
data = data.split()
data = " ".join(data)

print("Sample data:", data[:500])
print("Length of dataset:", len(data))

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Save tokenizer
pickle.dump(tokenizer, open("models/token.pkl", "wb"))

# Create sequences
sequence_data = tokenizer.texts_to_sequences([data])[0]
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary size:", vocab_size)

sequences = []
for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1]
    sequences.append(words)

print("Total sequences:", len(sequences))

sequences = np.array(sequences)

# Split X and y
X = sequences[:, 0:3]
y = sequences[:, 3]

# One-hot encode labels
y = to_categorical(y, num_classes=vocab_size)

print("Training data shape:", X.shape)
print("Labels shape:", y.shape)

# Build model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))
model.summary()

# Save best model checkpoint
checkpoint = ModelCheckpoint("models/next_words.h5", monitor="loss",
                             verbose=1, save_best_only=True)

model.compile(loss="categorical_crossentropy",
              optimizer=Adam(learning_rate=0.001))

# Train model
model.fit(X, y, epochs=70, batch_size=64, callbacks=[checkpoint])
