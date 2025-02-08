import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Connect to SQLite and fetch quiz history for a user
def fetch_user_performance(user_id):
    # migration... conn = sqlite3.connect("data/aprender_ai.db")
    conn = sqlite3.connect("backend/data/aprender_ai.db")

    query = """
        SELECT DATE(timestamp) AS quiz_date, COUNT(*) AS attempts,
               SUM(is_correct) AS correct_answers,
               (SUM(is_correct) * 100.0 / COUNT(*)) AS accuracy
        FROM quiz_logs
        WHERE user_id = ?
        GROUP BY quiz_date
        ORDER BY quiz_date ASC;
    """
    df = pd.read_sql_query(query, conn, params=(user_id,))
    conn.close()
    return df

# Plot accuracy trend over time
def plot_user_performance(user_id):
    df = fetch_user_performance(user_id)

    if df.empty:
        print(f"No quiz data found for User {user_id}.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["quiz_date"], df["accuracy"], marker="o", linestyle="-", color="b", label="Accuracy (%)")

    plt.xlabel("Date")
    plt.ylabel("Accuracy (%)")
    plt.title(f"User {user_id} Quiz Performance Over Time")
    plt.xticks(rotation=45)
    plt.ylim(0, 100)  # Accuracy should be between 0-100%
    plt.legend()
    plt.grid()

    # Save plot as image
    output_path = Path(f"data/user_{user_id}_quiz_progress.png")
    plt.savefig(output_path)
    plt.show()
    print(f"✅ Performance graph saved as '{output_path}'")

# Run visualization for a specific user
if __name__ == "__main__":
    user_id = int(input("Enter user ID to visualize performance: "))
    plot_user_performance(user_id)
