# 🛠 Aprender AI - Developer Setup & Installation Guide
(Follow these steps before using the Start-Up Guide.)

## 1️⃣ Clone the Repository
Navigate to your preferred development folder and run:

powershell
git clone <repository-url>
cd AprenderAI
(Replace <repository-url> with the actual GitHub link.)

## 2️⃣ Set Up Python Virtual Environment
Windows:
powershell
python -m venv venv
venv\Scripts\activate

Mac/Linux:
bash
python3 -m venv venv
source venv/bin/activate
✅ You should see (venv) in your terminal.

## 3️⃣ Install Backend Dependencies
powershell
pip install -r backend/requirements.txt
(Ensure you're inside the root project folder, not backend.)

## 4️⃣ Set Up the Database
Ensure the SQLite database exists:

powershell
dir backend/data/
If aprender_ai.db is missing, create the database schema manually:

powershell
python backend/setup_database.py
(If setup_database.py doesn’t exist, manually create backend/data/aprender_ai.db and add tables using DB Browser for SQLite.)

## 5️⃣ Train the AI Model (First-Time Setup)
Train the Decision Tree model before starting the system:

powershell
python backend/train_decision_tree.py
✅ Expected output:

csharp
Model Accuracy: 0.89
✅ Model successfully saved as 'backend/data/decision_tree_model.pkl'

## 6️⃣ Install Frontend Dependencies
Move into the frontend folder and install packages:

powershell
cd frontend
npm install
(Only needed the first time or after modifying package.json.)

## 7️⃣ Configure Environment Variables (If Needed)
If any API keys or secrets are required, create a .env file in the backend folder:

ini
DB_PATH=backend/data/aprender_ai.db
API_KEY=your-api-key-here
(Modify main.py to read from this .env file if required.)

## 8️⃣ Verify Installation
Check installed packages:

powershell
pip list  # Backend dependencies  
npm list --depth=0  # Frontend dependencies  

## 🎉 Setup Complete! Now, follow the Start-Up Guide (README.md) to launch Aprender AI! 