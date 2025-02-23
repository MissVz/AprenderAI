from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sqlite3
import random
import joblib
import numpy as np
import logging
from typing import Optional
from pydantic import BaseModel
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd

app = FastAPI()

# Enable CORS for the frontend at http://127.0.0.1:5173
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173"],  # Allow frontend to make requests
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Database connection function
def get_db_connection():
    # Path to the database
    conn = sqlite3.connect("./data/aprender_ai.db")
    conn.row_factory = sqlite3.Row  # Enables dictionary-like row access
    return conn

# Load the trained Decision Tree Model
model_path = "data/decision_tree_model.pkl"
clf = joblib.load(model_path)

# Initialize Q-table for reinforcement learning
Q_table = defaultdict(lambda: [0, 0, 0])  # Beginner, Intermediate, Advanced states

def predict_difficulty(attempts, accuracy):
    input_data = np.array([[attempts, accuracy]])
    return clf.predict(input_data)[0]  # Returns 0 (Beginner), 1 (Intermediate), 2 (Advanced)

def q_learning_adjust_difficulty(user_id, current_difficulty, reward):
    """
    Adjusts difficulty using Q-learning principles.
    Reward = 1 for correct, -1 for incorrect
    """
    learning_rate = 0.1
    discount_factor = 0.9

    # Choose an action based on Q-values (Easier, Same, Harder)
    action = np.argmax(Q_table[user_id]) if random.random() > 0.2 else random.choice([0, 1, 2])

    # Update Q-value for the user
    Q_table[user_id][current_difficulty] = (1 - learning_rate) * Q_table[user_id][current_difficulty] + \
                                           learning_rate * (reward + discount_factor * max(Q_table[user_id]))

    # Adjust difficulty based on chosen action
    difficulty_adjustment = [-1, 0, 1]  # Easier, Same, Harder
    new_difficulty = max(0, min(2, current_difficulty + difficulty_adjustment[action]))

    return new_difficulty

@app.get("/quiz/{user_id}")
def get_adaptive_quiz(user_id: int):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        print(f"DEBUG: Checking user progress for user_id={user_id}")

        # Fetch user performance data
        cursor.execute("""
            SELECT COUNT(*) AS attempts, SUM(is_correct) AS correct_answers
            FROM user_progress
            WHERE user_id = ?
        """, (user_id,))
        
        user_data = cursor.fetchone()
        print(f"DEBUG: Retrieved user data -> {user_data}")  # Debugging log

        if not user_data or user_data["attempts"] == 0:
            predicted_difficulty = 0  # Default to Beginner
        else:
            attempts = user_data["attempts"]
            correct_answers = user_data["correct_answers"]
            accuracy = correct_answers / attempts

            # Predict difficulty using trained model
            predicted_difficulty = int(predict_difficulty(attempts, accuracy))  # ✅ Convert to Python int
            predicted_difficulty = int(q_learning_adjust_difficulty(user_id, predicted_difficulty, reward=1 if accuracy > 0.7 else -1))

        print(f"DEBUG: User {user_id} classified as {predicted_difficulty}")

        # Fetch quiz question
        cursor.execute("SELECT english, spanish FROM translations ORDER BY RANDOM() LIMIT 1;")
        word_pair = cursor.fetchone()

        if not word_pair:
            raise ValueError("No quiz questions found in database!")

        correct_answer = word_pair["spanish"]

        # Fetch 3 incorrect answers
        cursor.execute("SELECT spanish FROM translations WHERE spanish != ? ORDER BY RANDOM() LIMIT 3;", (correct_answer,))
        incorrect_answers = [row["spanish"] for row in cursor.fetchall()]

        while len(incorrect_answers) < 3:
            incorrect_answers.append("N/A")  # Placeholder if dataset is limited

        choices = incorrect_answers + [correct_answer]
        random.shuffle(choices)

        conn.close()

        return {
            "user_id": user_id,
            "predicted_difficulty": predicted_difficulty,  # ✅ Now a Python int, not numpy.int64
            "question": f"What is the Spanish translation for '{word_pair['english']}'?",
            "choices": choices,
            "answer": correct_answer  # Remove in production for security
        }

    except Exception as e:
        print(f"❌ ERROR in /quiz/{user_id}: {str(e)}")
        return {"error": "An internal server error occurred", "details": str(e)}, 500

    finally:
        conn.close()


class QuizResponse(BaseModel):
    user_id: int
    question: str
    user_answer: str
    correct_answer: Optional[str] = "Unknown"  # Make this optional

@app.post("/quiz/submit")
def submit_quiz_response(response: QuizResponse):
    try:
        print(f"DEBUG: Received answer submission - User ID: {response.user_id}, Question: {response.question}, "
              f"User Answer: {response.user_answer}, Correct Answer: {response.correct_answer}")

        is_correct = response.user_answer == response.correct_answer

        conn = get_db_connection()
        cursor = conn.cursor()

        # Insert response into user_progress
        cursor.execute("""
            INSERT INTO user_progress (user_id, question, user_answer, correct_answer, is_correct)
            VALUES (?, ?, ?, ?, ?);
        """, (response.user_id, response.question, response.user_answer, response.correct_answer, is_correct))

        conn.commit()

        # Fetch next question **after commit to ensure database consistency**
        cursor.execute("SELECT english, spanish FROM translations ORDER BY RANDOM() LIMIT 1;")
        word_pair = cursor.fetchone()

        correct_answer = word_pair["spanish"]

        # Fetch 3 incorrect answers
        cursor.execute("SELECT spanish FROM translations WHERE spanish != ? ORDER BY RANDOM() LIMIT 3;", (correct_answer,))
        incorrect_answers = [row["spanish"] for row in cursor.fetchall()]

        # Ensure 3 incorrect answers
        while len(incorrect_answers) < 3:
            incorrect_answers.append("N/A")  # Placeholder if dataset is limited

        # Shuffle choices
        choices = incorrect_answers + [correct_answer]
        random.shuffle(choices)

        conn.close()

        return {
            "message": "Response recorded",
            "correct": is_correct,
            "next_question": {
                "question": f"What is the Spanish translation for '{word_pair['english']}'?",
                "choices": choices,
                "answer": correct_answer  # Remove in production for security
            }
        }

    except Exception as e:
        print(f"❌ Error in /quiz/submit: {str(e)}")
        return {"error": "An internal server error occurred"}, 500

    finally:
        conn.close()


@app.get("/quiz/difficulty_trend/{user_id}")
async def plot_difficulty_trend(user_id: int):
    # return {"message": f"Difficulty trend for user {user_id} will be plotted here."}
    try:
        conn = sqlite3.connect("data/aprender_ai.db")
        query = """
            SELECT DATE(timestamp) AS quiz_date, COUNT(*) AS attempts,
                   SUM(is_correct) AS correct_answers,
                   (SUM(is_correct) * 100.0 / COUNT(*)) AS accuracy
            FROM user_progress
            WHERE user_id = ?
            GROUP BY quiz_date
            ORDER BY quiz_date ASC;
        """
        df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()

        if df.empty:
            return {"message": "No quiz data found for this user."}

        # Plot difficulty trend
        plt.figure(figsize=(10, 6))
        plt.plot(df["quiz_date"], df["accuracy"], marker="o", linestyle="-", color="b", label="Accuracy (%)")
        plt.xlabel("Date")
        plt.ylabel("Accuracy (%)")
        plt.title(f"User {user_id} Quiz Performance Over Time")
        
        # Improve bottom axis label visibility
        plt.xticks(rotation=45, ha="right", fontsize=8)  # Rotate and align right
        plt.ylim(0, 100)  # Accuracy should be between 0-100%
        plt.legend()
        plt.grid()

        # Save and return image path
        image_path = f"data/user_{user_id}_difficulty_trend.png"
        plt.savefig(image_path)
        plt.close()
        return FileResponse(image_path, media_type="image/png")

    except Exception as e:
        print(f"❌ ERROR in /quiz/difficulty_trend/{user_id}: {str(e)}")
        return {"error": "An internal server error occurred", "details": str(e)}, 500