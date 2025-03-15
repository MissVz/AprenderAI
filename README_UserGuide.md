# 📖 Aprender AI - User Guide
(Available API & Frontend URLs for Users and Developers)

# 🌐 1️⃣ Frontend - User Interface
📌 Open the Aprender AI App
🔗 URL: http://localhost:5173/

This is the main user interface where you can take adaptive Spanish vocabulary quizzes.
Enter your User ID, answer questions, and get feedback.

# ⚙️ 2️⃣ Backend - FastAPI Server
📌 API Documentation (Swagger UI)
🔗 URL: http://127.0.0.1:8000/docs

Provides an interactive API explorer to test endpoints.
Allows you to send API requests without writing code.

📌 API Endpoints
All API endpoints follow this base URL:
🔗 http://127.0.0.1:8000/

Method	Endpoint	Description
GET	/quiz/{user_id}	Fetches an adaptive quiz question for the given User ID.
POST	/quiz/submit	Submits an answer and updates user progress.
GET	/quiz/difficulty_trend/{user_id}	Returns difficulty trends over time for the given User ID.

## 📝 Example Usage
Get a quiz question for User ID 1
🔗 http://127.0.0.1:8000/quiz/1
Submit an answer (JSON request needed)
json
Copy
Edit
{
  "user_id": 1,
  "question": "What is the Spanish word for apple?",
  "user_answer": "manzana",
  "correct_answer": "manzana"
}
View difficulty trends for User ID 1
🔗 http://127.0.0.1:8000/quiz/difficulty_trend/1

# 📊 3️⃣ Admin & Developer Tools
📌 Database Viewer (SQLite Browser)
Open backend/data/aprender_ai.db in DB Browser for SQLite.
Allows manual inspection of quiz logs, user progress, and translations.
📌 Model Training
Run python backend/train_decision_tree.py to update the AI model.

# 🚀 Summary of Available URLs
Type	URL	Purpose
Frontend	http://localhost:5173/	Take quizzes in the UI
API Docs	http://127.0.0.1:8000/docs	Test API requests interactively
Get Quiz	http://127.0.0.1:8000/quiz/{user_id}	Fetch a quiz question
Submit Answer	http://127.0.0.1:8000/quiz/submit	Send quiz answers to API
View Trends	http://127.0.0.1:8000/quiz/difficulty_trend/{user_id}	See quiz performance over time

## 🎉 Now you're ready to explore Aprender AI!