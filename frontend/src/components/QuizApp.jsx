import React, { useState, useEffect } from "react";
import "./QuizApp.css"; // Import CSS file for styling

function QuizApp() {
  const [quizData, setQuizData] = useState(null);
  const [error, setError] = useState(null);
  const [userId, setUserId] = useState(1);
  const [selectedAnswer, setSelectedAnswer] = useState(null);
  const [submissionResult, setSubmissionResult] = useState(null); // ✅ Ensure this is initialized
  const [isCorrect, setIsCorrect] = useState(null);

  useEffect(() => {
    console.log("Fetching quiz data for user", userId);
    setIsCorrect(null);  // ✅ Reset isCorrect when user ID changes
    fetch(`http://127.0.0.1:8000/quiz/${userId}`)
      .then(response => {
        if (!response.ok) {
          throw new Error("Server error, try again later.");
        }
        return response.json();
      })
      .then(data => setQuizData(data))
      .catch(error => setError(error.message));
  }, [userId]);

  const submitAnswer = () => {
    if (!selectedAnswer) return;
  
    console.log("Submitting answer:", {
      user_id: userId,
      question: quizData.question,
      user_answer: selectedAnswer,
      correct_answer: quizData.answer ?? "Unknown",  // ✅ Ensure a valid value
    });
  
    fetch("http://127.0.0.1:8000/quiz/submit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: Number(userId),
        question: quizData.question,
        user_answer: selectedAnswer,
        correct_answer: quizData.answer ?? "Unknown",  // ✅ Always send a valid answer
      }),
    })
      .then(response => response.json())
      .then(data => {
        console.log("Answer response:", data);
        setSubmissionResult(data.correct ? "✅ Correct!" : "❌ Incorrect!");
        setTimeout(() => {
          setSelectedAnswer(null);
          setSubmissionResult(null);
          setQuizData(null);
        }, 2000);
      })
      .catch(error => {
        console.error("Error submitting quiz:", error);
        setSubmissionResult("⚠️ Submission Failed. Try Again.");
      });
  };
  

  if (error) return <p style={{ color: "red" }}>{error}</p>;
  if (!quizData) return <p>Loading quiz...</p>;

  return (
    <div className="quiz-container">
      <h1><img src="/MexicanAIpersonalizedLearningAssistant_Logo.png" alt="Aprender AI - Interactive Spanish Quiz" className="quiz-logo" /></h1>
      <h2>Enter your User ID:</h2>
      <input
        type="number"
        className="user-input"
        value={userId}
        onChange={(e) => setUserId(e.target.value)}
      />
      <h2 className="question">{quizData.question}</h2>
      <ul className="answer-list">
        {quizData.choices.map((choice, index) => (
          <li key={index}>
            <button 
              className={`answer-button ${selectedAnswer === choice ? (isCorrect === true ? "correct" : isCorrect === false ? "incorrect" : "selected") : ""}`}
              onClick={() => setSelectedAnswer(choice)}
            >
              {choice}
            </button>
          </li>
        ))}
      </ul>
      {submissionResult && <p className="submission-result">{submissionResult}</p>}
      <button className="submit-button" onClick={submitAnswer} disabled={!selectedAnswer}>Submit Answer</button>
    </div>
  );
}

export default QuizApp;