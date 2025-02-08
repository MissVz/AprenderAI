import sqlite3

# Connect to the database
conn = sqlite3.connect("data/aprender_ai.db")
cursor = conn.cursor()

# Fetch and display user progress data
cursor.execute("SELECT * FROM user_progress LIMIT 5;")
rows = cursor.fetchall()

for row in rows:
    print(row)

conn.close()
