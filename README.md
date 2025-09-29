✨ New Word Predictor ✨
🚀 A Deep Learning based NLP project that predicts the next word in a sentence using LSTMs.

🌟 Features:
✅ Built with TensorFlow & Keras
✅ Trained on Pride and Prejudice (Jane Austen) 📖
✅ Interactive prediction loop 🔮
✅ Saves and reuses tokenizer for future predictions 💾
✅ Clean modular structure (train & predict scripts) ⚡

🛠️ Tech Stack:
🐍 Python 3
🧠 TensorFlow / Keras
🔢 NumPy
📦 Pickle

📂 Project Structure
📦 nextword
 ┣ 📂 data
 ┃ ┗ 📜 Pride_and_Prejudice.txt      # Training dataset
 ┣ 📂 src
 ┃ ┣ 📜 train.py                     # Training script
 ┃ ┗ 📜 predict.py                   # Prediction script
 ┣ 📜 requirements.txt               # Python dependencies
 ┣ 📜 README.md                      # Project documentation

⚙️ How It Works:
Preprocessing 🧹
Cleans text data (removes extra symbols, spaces).
Tokenizes text & generates word sequences.
Model Training 🏋️
Embedding → LSTM (×2) → Dense → Softmax.
Optimized with Adam + categorical crossentropy.

Prediction 🔮
Input: Last 3 words of a sentence.
Output: Predicted next word.

▶️ Usage
1️⃣ Clone the repo
git clone https://github.com/chvarun-stack/new_word_predictor.git
cd new_word_predictor

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Train the model
python src/train.py

4️⃣ Run predictions
python src/predict.py

📊 Example Output
Enter your line: I love to
Predicted word: read

🔮 Future Improvements
🌍 Train on larger, diverse datasets.
🌐 Deploy as a Flask/Django web app.
⚡ Optimize model for faster inference.

