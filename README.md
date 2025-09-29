# New Word Predictor

A Deep Learning–based NLP project that predicts the **next word** in a sentence using **LSTMs**.

---

## Features
- Built with **TensorFlow** & **Keras**  
- Trained on *Pride and Prejudice* (Jane Austen)  
- Interactive prediction loop  
- Saves and reuses tokenizer for future predictions  
- Modular structure (separate training & prediction scripts)  

---

## Tech Stack
- Python 3  
- TensorFlow / Keras  
- NumPy  
- Pickle  

---

## Project Structure
new_word_predictor/
├── data/
│ └── Pride_and_Prejudice.txt # Training dataset
├── src/
│ ├── train.py # Training script
│ ├── predict.py # Prediction script
│ └── requirements.txt # Dependencies
└── README.md # Documentation



---

## How It Works

1. **Preprocessing**  
   - Cleans text data (removes extra symbols, spaces).  
   - Tokenizes text & generates word sequences.  

2. **Model Training**  
   - Embedding → LSTM (×2) → Dense → Softmax  
   - Optimized with Adam + categorical crossentropy  

3. **Prediction**  
   - **Input**: Last 3 words of a sentence  
   - **Output**: Predicted next word  

---

## Usage

1. Clone the repository:
   

   git clone https://github.com/chvarun-stack/new_word_predictor.git

   cd new_word_predictor
   
2.Install dependencies:

pip install -r requirements.txt

3.Train the model:

python src/train.py

4.Run predictions:

python src/predict.py

5.Example

Enter your line: I love to

Predicted word: read
