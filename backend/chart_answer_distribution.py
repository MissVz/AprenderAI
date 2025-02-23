import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect("data/aprender_ai.db")

# Fetch correct vs. incorrect answers
query = """
SELECT SUM(is_correct) AS correct_answers, COUNT(*) - SUM(is_correct) AS incorrect_answers
FROM user_progress
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Extract values
correct_count = df["correct_answers"].iloc[0]
incorrect_count = df["incorrect_answers"].iloc[0]

# Create Pie Chart
plt.figure(figsize=(5, 5))
plt.pie(
    [correct_count, incorrect_count], 
    labels=["Correct", "Incorrect"], 
    autopct="%1.1f%%", 
    colors=["green", "red"], 
    startangle=90
)
plt.title("Overall Correct vs. Incorrect Answers")

# Save and Show Plot
plt.savefig("data/correct_vs_incorrect.png")
plt.show()
