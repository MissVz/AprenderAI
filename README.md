# AprenderAI
AI610 Agent Based Systems: Aprender AI is an adaptive Spanish vocabulary learning assistant using Q-Learning for real-time quiz difficulty adjustment and Decision Trees for preprocessing. Built with FastAPI, React.js, and SQLite, it tracks user progress and optimizes learning pathways dynamically. 🚀

# 📌 Aprender AI Start-Up Cheat Sheet
(Adaptive Spanish Vocabulary Quiz with AI-Powered Personalization)

## 1️⃣ Activate the Virtual Environment
Before running any commands, ensure the Python virtual environment is activated.

powershell
cd path\to\AprenderAI-Teagan  # Navigate to project root
venv\Scripts\activate  # Activate virtual environment (Windows)
(For macOS/Linux, use: source venv/bin/activate)

## 2️⃣ Start the Backend (FastAPI)
Move into the backend folder and start the API:

powershell
cd backend
uvicorn main:app --reload
✅ If successful, API is running at:
👉 http://127.0.0.1:8000/

✅ Test API Endpoints:

Interactive Docs: http://127.0.0.1:8000/docs
Fetch a quiz question: http://127.0.0.1:8000/quiz/1

## 3️⃣ Start the Frontend (React)
In a new terminal, navigate to the frontend folder:

powershell
cd ../frontend
npm install  # (Only needed the first time)
npm run dev  # Start frontend
✅ If successful, open the app at:
👉 http://localhost:5173/

4️⃣ Train the AI Model (If Not Trained)
(Skip this if already trained!)
In the backend folder, run:

powershell
python train_decision_tree.py
✅ Expected output:

csharp
Model Accuracy: 0.89
✅ Model successfully saved as 'data/decision_tree_model.pkl'

## 5️⃣ Use the App!
Take a Quiz
Open http://localhost:5173/
Enter a User ID (e.g., 1)
Answer the quiz questions
The system will adjust difficulty dynamically!

## 6️⃣ Debugging & Logs
Check Backend Logs
If something isn’t working, check the FastAPI logs:

powershell
uvicorn main:app --reload --log-level debug
Restart Everything (If Needed)
Close & restart your terminal.
Ensure venv is activated.
Restart backend & frontend.

## 7️⃣ Shutdown Instructions
To stop Aprender AI:
Stop Backend: Press CTRL + C in the backend terminal.
Stop Frontend: Press CTRL + C in the frontend terminal.

# ⚡ Quick Start Recap ⚡
Step	Command
1. Activate venv	venv\Scripts\activate
2. Start Backend	cd backend && uvicorn main:app --reload
3. Start Frontend	cd ../frontend && npm run dev
4. Train AI Model	python backend/train_decision_tree.py
5. Open App	http://localhost:5173/
6. Test API	http://127.0.0.1:8000/docs

🎉 You’re ready to go! 