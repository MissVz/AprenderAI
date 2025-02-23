import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect("data/aprender_ai.db")

# Fetch time to mastery data
query = """
SELECT question, MIN(attempt_number) AS attempts_before_correct
FROM (
    SELECT question, user_id, timestamp, 
           COUNT(*) OVER (PARTITION BY user_id, question ORDER BY timestamp) AS attempt_number,
           is_correct
    FROM user_progress
) WHERE is_correct = 1
GROUP BY question
ORDER BY attempts_before_correct DESC
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Create Bar Chart
plt.figure(figsize=(10, 5))
df.sort_values(by="attempts_before_correct", ascending=False).plot(
    kind="bar", x="question", y="attempts_before_correct", color="blue", alpha=0.7, legend=False
)

# Customize chart appearance
plt.xlabel("Question", fontsize=12)
plt.ylabel("Attempts Before First Correct Answer", fontsize=12)
plt.title("Time to Mastery: Attempts Per Question", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.subplots_adjust(bottom=0.35)  # Adjust bottom margin for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save and Show Plot
plt.tight_layout()
plt.savefig("data/time_to_mastery.png")
plt.show()
