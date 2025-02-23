import sqlite3
import pandas as pd
import joblib
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Ensure the directory exists
model_dir = Path("data")  # Change from "backend/data" to "data"
model_dir.mkdir(parents=True, exist_ok=True)

model_path = model_dir / "decision_tree_model.pkl"

# Connect to SQLite and fetch training data
def load_data():
    conn = sqlite3.connect("data/aprender_ai.db")
    query = """
    SELECT question, COUNT(*) AS attempts,
           SUM(is_correct) AS correct_answers
    FROM user_progress
    GROUP BY question;
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Add a new column: accuracy rate
    df["accuracy"] = df["correct_answers"] / df["attempts"]

    # Label difficulty based on accuracy
    def classify_difficulty(accuracy):
        if accuracy < 0.6:
            return "Beginner"
        elif 0.6 <= accuracy < 0.80:
            return "Intermediate"
        else:
            return "Advanced"

    df["difficulty"] = df["accuracy"].apply(classify_difficulty)
    
    return df

# Load dataset
df = load_data()

# Encode difficulty labels
difficulty_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
df["difficulty_encoded"] = df["difficulty"].map(difficulty_map)

# Features and labels
X = df[["attempts", "accuracy"]]
y = df["difficulty_encoded"]

# Ensure enough data for training
if len(df) < 5:
    print(f"Not enough data to train the model. Only {len(df)} samples available.")
else:
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Decision Tree Classifier with optimized parameters
    clf = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    clf.fit(X_train, y_train)

    # Test model accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {accuracy:.2f}")

    # ✅ Save the trained model
    joblib.dump(clf, model_path)

    # Verify if the model was saved
    if Path(model_path).exists():
        print(f"✅ Model successfully saved as '{model_path}'")
    else:
        print("❌ Model failed to save!")
