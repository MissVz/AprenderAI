import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the SQLite database
conn = sqlite3.connect("data/aprender_ai.db")

# Load user progress data
query = """
    SELECT user_id, DATE(timestamp) as quiz_date, is_correct, question
    FROM user_progress
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Convert timestamp to datetime format
df["quiz_date"] = pd.to_datetime(df["quiz_date"])

### 1️⃣ Difficulty Trend Over Time (Line Chart)
# Simulated difficulty mapping (this should be retrieved from your AI model if available)
difficulty_map = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

# Assign difficulty levels (assuming users progress over time)
df["difficulty_level"] = df.groupby("user_id")["is_correct"].cumsum() % 3  # Simulated

plt.figure(figsize=(12, 6))
for user in df["user_id"].unique():
    user_df = df[df["user_id"] == user]
    plt.plot(user_df["quiz_date"], user_df["difficulty_level"], marker="o", linestyle="-", label=f"User {user}")

plt.xlabel("Date")
plt.ylabel("Difficulty Level (0=Beginner, 2=Advanced)")
plt.title("User Difficulty Progression Over Time")
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("data/difficulty_trend.png")
plt.show()


### 2️⃣ Correct vs. Incorrect Answers (Pie Chart)
correct_count = df["is_correct"].sum()
incorrect_count = len(df) - correct_count

plt.figure(figsize=(4, 4))
plt.pie(
    [correct_count, incorrect_count], 
    labels=["Correct", "Incorrect"], 
    autopct="%1.1f%%", 
    colors=["green", "red"], 
    startangle=90
)
plt.title("Overall Correct vs. Incorrect Answers")
plt.savefig("data/correct_vs_incorrect.png")
plt.show()


### 3️⃣ Time to Mastery (Bar Chart)
# Count attempts per question until first correct response
df_sorted = df.sort_values(["user_id", "question", "quiz_date"])
df_sorted["attempt_number"] = df_sorted.groupby(["user_id", "question"]).cumcount() + 1
df_first_correct = df_sorted[df_sorted["is_correct"] == 1].groupby("question")["attempt_number"].min()

plt.figure(figsize=(6, 3))
df_first_correct.sort_values().plot(kind="bar", color="blue", alpha=0.7)
plt.xlabel("Question")
plt.ylabel("Attempts Before First Correct Answer")
plt.title("Time to Mastery: Attempts Per Question")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("data/time_to_mastery.png")
plt.show()
