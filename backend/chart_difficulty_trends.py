import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to SQLite database
conn = sqlite3.connect("data/aprender_ai.db")

# Fetch difficulty trend data
query = """
SELECT u.name, u.level, up.user_id, DATE(up.timestamp) as quiz_date, 
       SUM(up.is_correct) AS correct_answers, COUNT(*) AS attempts
FROM user_progress up
JOIN users u ON up.user_id = u.id
GROUP BY up.user_id, quiz_date
ORDER BY quiz_date ASC
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Calculate accuracy
df["accuracy"] = (df["correct_answers"] / df["attempts"]) * 100

# Plot Line Chart with Improved Legend
plt.figure(figsize=(8, 5))
for user_id in df["user_id"].unique():
    user_df = df[df["user_id"] == user_id]
    user_name = user_df["name"].iloc[0]  # Fetch the first occurrence of name
    user_level = user_df["level"].iloc[0]  # Fetch user level
    
    plt.plot(
        user_df["quiz_date"], user_df["accuracy"], marker="o", linestyle="-",
        label=f"{user_name} ({user_level})"  # Legend includes name and level
    )

# Customize chart appearance
plt.xlabel("Date", fontsize=12)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("User Quiz Accuracy Over Time", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.legend(title="Users", fontsize=10)  # Adjust legend title and font size
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show or save the figure
plt.tight_layout()
plt.savefig("data/difficulty_trend_with_names.png")
plt.show()
