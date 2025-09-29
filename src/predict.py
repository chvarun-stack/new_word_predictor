# predict.py
# ----------------------------------------
# Next Word Prediction - Prediction Script
# ----------------------------------------

import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load trained model and tokenizer
model = load_model("models/next_words.h5")
tokenizer = pickle.load(open("models/token.pkl", "rb"))

# Prediction function
def Predict_Next_Words(model, tokenizer, text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)
    
    preds = np.argmax(model.predict(sequence))
    predicted_word = ""

    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break
    
    return predicted_word

# Interactive loop
if __name__ == "__main__":
    while True:
        text = input("Enter your line (type 0 to exit): ")
        if text == "0":
            print("Execution completed.")
            break
        else:
            try:
                words = text.split(" ")
                words = words[-3:]  # last 3 words
                next_word = Predict_Next_Words(model, tokenizer, words)
                print("Predicted word:", next_word)
            except Exception as e:
                print("Error occurred:", e)
                continue
