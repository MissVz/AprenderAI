import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Connect to the SQLite database
db_path = "data/aprender_ai.db"  # Update if needed
conn = sqlite3.connect(db_path)

# Load user progress data
query = """
    SELECT question, COUNT(*) AS attempts, SUM(is_correct) AS correct_answers
    FROM user_progress
    GROUP BY question;
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Calculate accuracy rate
df["accuracy"] = df["correct_answers"] / df["attempts"]

# Define function to categorize questions based on content
def categorize_question(question):
    if any(word in question.lower() for word in ["color", "red", "blue", "green"]):
        return "Basic Colors"
    elif any(word in question.lower() for word in ["hello", "goodbye", "thank", "please"]):
        return "Common Phrases"
    elif any(word in question.lower() for word in ["one", "two", "three", "four"]):
        return "Numbers"
    elif any(word in question.lower() for word in ["monday", "tuesday", "wednesday"]):
        return "Days of the Week"
    elif any(word in question.lower() for word in ["apple", "banana", "bread", "cheese"]):
        return "Food Items"
    elif len(question.split()) < 4:
        return "Simple Sentences"
    elif len(question.split()) < 7:
        return "Intermediate Phrases"
    elif len(question.split()) < 10:
        return "Complex Sentences"
    else:
        return "Advanced Grammar"

# Apply categorization
df["Category"] = df["question"].apply(categorize_question)

# Group data by category and calculate average attempts
df_grouped = df.groupby("Category")["attempts"].mean().reset_index()

# Create a bar chart (landscape format for clarity)
plt.figure(figsize=(12, 5))
plt.bar(df_grouped["Category"], df_grouped["attempts"], color="royalblue")

# Labels and title
plt.xlabel("Question Category", fontsize=12)
plt.ylabel("Avg Attempts Before Mastery", fontsize=12)
plt.title("Attempts Required for Mastery by Category", fontsize=14)
plt.xticks(rotation=20, ha="right")  # Tilt x-axis labels for readability

# Save the figure
plt.savefig("data/attempts_required_mastery_real_data.png", bbox_inches="tight")
plt.show()
