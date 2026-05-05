# Gesture Quiz 🎯
An AI-powered quiz application that allows users to answer questions using hand gestures instead of traditional input methods. The system uses computer vision and deep learning to recognize gestures in real-time through a webcam.

## 🚀 Features
- ✋ Gesture-based answer selection (A, B, C, D)
- 🎥 Real-time webcam integration
- 🧠 Deep learning model for gesture recognition
- ⏱ Timed questions
- 📊 Topic-wise performance analysis
- 🎯 Weak topic identification

## 🧠 Gesture Controls
- A → 1 Finger  
- B → 2 Fingers / Peace ✌️  
- C → C-shaped hand  
- D → Yo sign (🤙)  
- None → Back of hand (no selection)

## 🛠️ Tech Stack
- Python
- OpenCV
- PyTorch
- NumPy

## 📂 Project Structure
GAME/
│── main_quiz.py # Main application
│── models/
│ ├── gesture_net.py # Model architecture
│ └── best_model.pth # Trained model
│── data/
│ └── questions.json # Quiz questions
│── utils/ # Helper functions
│── train_model.py # Model training script
│── test.py # Testing script


## ▶️ How to Run
pip install torch torchvision opencv-python numpy
python main_quiz.py
### 💡 Use Case
This project demonstrates a touchless interaction system, useful for:
Smart classrooms
Accessibility applications
Interactive AI systems

### 📌 Future Improvements
Improved UI design
Gesture confidence visualization
Leaderboard system
Mobile deployment

### 👩‍💻 Authors
Jasman Kaur
Jivitesh Bansal
Sanchit Luthra
![Demo](demo.gif)

👉 Upload a short screen recording → convert to GIF → add here
