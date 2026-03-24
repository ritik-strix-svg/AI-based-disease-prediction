
# 🧠 AI Disease Prediction System

A full-stack AI-powered web application that predicts the risk of **Diabetes** and **Heart Disease** using machine learning models. The system provides secure authentication, role-based access (User/Admin), and stores prediction history for analysis.

---

## 🚀 Features

### 🔐 Authentication & Security
- JWT-based authentication
- Role-based access control (User & Admin)
- Secure password hashing using bcrypt
- Protected API routes

### 👤 User Features
- Login & Signup system
- Predict Diabetes using 8 medical parameters
- Predict Heart Disease using 11 clinical features
- View prediction results with probability
- Access personal prediction history
- Responsive and interactive UI

### 👑 Admin Features
- Admin dashboard with analytics
- View all user predictions
- Monitor system usage
- Manage users

### 🤖 Machine Learning
- Integrated trained ML models (Scikit-learn)
- Real-time prediction using joblib
- Probability-based output

### 🗄 Database
- MySQL database integration
- Tables:
  - Users
  - Predictions
- Stores:
  - User details
  - Prediction history
  - Timestamp tracking

---

## 🛠 Tech Stack

**Frontend:**
- HTML, CSS, JavaScript

**Backend:**
- Python (Flask)
- Flask-JWT-Extended
- Flask-Bcrypt

**Machine Learning:**
- Scikit-learn
- Joblib

**Database:**
- MySQL

---

## 📊 System Architecture

- Frontend sends input → Flask API
- Backend processes data using ML model
- Prediction result returned with probability
- Data stored in MySQL
- JWT used for secure communication

---

## ⚙️ Installation

1. Clone the repository:
```bash
git clone https://github.com/ritik-strix-svg/ai-disease-prediction.git
cd ai-disease-prediction
