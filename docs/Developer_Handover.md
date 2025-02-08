# **Developer Handover Guide: AI-Powered Adaptive Learning System**

## **Project Overview**
This project is an AI-powered **Adaptive Spanish Vocabulary Learning Assistant** that dynamically adjusts quiz difficulty based on user performance. It is built using:

- **Frontend:** React.js (Interactive quiz UI)
- **Backend:** FastAPI (Manages AI model requests and user interactions)
- **Database:** SQLite (Stores user progress, quiz data, and AI training data)
- **AI Models:**
  - **Decision Trees** (Preprocessing for difficulty classification)
  - **Q-Learning** (Reinforcement learning for real-time quiz difficulty adjustment)
  - **Artificial Neural Networks (ANNs)** (**To be implemented**)  
- **Visualization:** Matplotlib (Plots user progress trends)

This guide will help the next developer set up the project and implement ANNs for improved learning adaptation.

---

## **1️⃣ Setup Instructions**
### **1.1 Clone the Repository & Set Up the Environment**
```sh
# Clone the GitHub repository
git clone https://github.com/YOUR_REPO_URL/AI610-Adaptive-Quiz.git
cd AI610-Adaptive-Quiz
```

### **1.2 Backend Setup (FastAPI & Dependencies)**
```sh
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### **1.3 Frontend Setup (React.js)**
```sh
cd frontend
npm install
npm run dev
```

- **Key Frontend Files:**
  - `App.jsx` → Main React component
  - `main.jsx` → React entry point
  - `QuizApp.jsx` → Handles quiz UI
  - **`QuizApp.css`** → Styles the quiz interface

✅ **Ensure `QuizApp.css` is correctly imported in `QuizApp.jsx`.**
```sh
cd frontend
npm install
npm run dev
```

### **1.4 Database Initialization (SQLite)**
```sh
cd backend
sqlite3 data/aprender_ai.db
.tables  # Verify database structure
```
If needed, run:
```sql
CREATE TABLE IF NOT EXISTS user_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    user_answer TEXT NOT NULL,
    correct_answer TEXT NOT NULL,
    is_correct BOOLEAN NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## **2️⃣ Key Backend Components**
### **2.1 Main API (`backend/main.py`)**
Handles quiz generation, user responses, and AI model integration.
```sh
uvicorn main:app --reload
```
- **Key Endpoints:**
  - `GET /quiz/{user_id}` → Fetches a quiz question based on user difficulty level.
  - `POST /quiz/submit` → Saves user quiz responses and updates learning progress.
  - `GET /quiz/logs/{user_id}` → Retrieves user quiz logs.
  - `GET /quiz/difficulty_trend/{user_id}` → Returns a difficulty trend graph.

### **2.2 AI Models (`backend/train_decision_tree.py`)**
Trains and saves **Decision Tree models** for preprocessing difficulty classification.
```sh
python train_decision_tree.py
```

### **2.3 Reinforcement Learning (`backend/main.py`)**
Q-Learning dynamically adjusts difficulty based on user responses.
```python
def q_learning_adjust_difficulty(user_id, current_difficulty, reward):
    learning_rate = 0.1
    discount_factor = 0.9
    action = np.argmax(Q_table[user_id]) if random.random() > 0.2 else random.choice([0, 1, 2])
    Q_table[user_id][current_difficulty] = (1 - learning_rate) * Q_table[user_id][current_difficulty] + \
        learning_rate * (reward + discount_factor * max(Q_table[user_id]))
    return max(0, min(2, current_difficulty + [-1, 0, 1][action]))
```

---

## **3️⃣ Next Steps: Implement Artificial Neural Networks (ANNs)**
### **3.1 Objective of ANN Integration**
The ANN will **predict learning weaknesses** and recommend customized quizzes based on past responses.

### **3.2 Steps to Implement ANN**
#### **Step 1: Extract Quiz Logs for ANN Training**
Modify `convert_user_progress_to_logs.py` to structure data for ANN training.
```sh
python convert_user_progress_to_logs.py
```

#### **Step 2: Develop ANN Model (`backend/train_ann.py`)**
Create a new script for ANN model training:
```python
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("data/quiz_logs.csv")
X = df[["attempts", "accuracy"]]
y = df["difficulty"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define ANN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Compile & train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("data/ann_model.h5")
```

#### **Step 3: Integrate ANN into FastAPI (`main.py`)**
Modify `/quiz/{user_id}` to use ANN predictions:
```python
import tensorflow as tf
model = tf.keras.models.load_model("data/ann_model.h5")

def predict_ann_difficulty(features):
    features = scaler.transform([features])
    prediction = model.predict(features)
    return np.argmax(prediction)
```

#### **Step 4: Test ANN-Based Quiz Adjustments**
Restart FastAPI and verify quiz difficulty adjustments:
```sh
uvicorn main:app --reload
```

---

## **4️⃣ Final Checklist Before ANN Implementation**
✅ Ensure database contains enough quiz logs for ANN training.
✅ Verify `backend/data/quiz_logs.csv` is structured correctly.
✅ Update `backend/main.py` to call ANN predictions.
✅ Test quiz difficulty adjustments after ANN integration.

---

## **📌 Conclusion**
This guide provides **a full setup, code structure, and next steps for implementing ANN integration** in the AI-powered adaptive learning system.