✈️ Pilot Drowsiness Detection System

A real-time AI-based system to detect driver/pilot drowsiness using webcam video.

---

🚀 Features

- Detects Drowsiness, Inattentiveness, and Eye Closure
- Uses CNN + LSTM model
- MediaPipe-based face & eye tracking
- Works in low-light conditions
- Real-time webcam monitoring

---

🛠️ Tech Stack

- Python
- PyTorch
- OpenCV
- MediaPipe
- Streamlit

---

▶️ How to Run

1. Clone the repository:

git clone
https://github.com/NamanSaxena2029/Pilot-Drowsiness.git
cd pilot-drowsiness-detection

2. Install dependencies:

pip install -r requirements.txt

3. Run the app:

streamlit run app.py

---

pilot-drowsiness/
├── app.py
├── drowsy_detection.py
├── step1_recrop.py
├── step2_retrain.py
├── step1_recrop_lowlight.py
├── step2_finetune_ll.py
│
├── best_model_ll.pth
├── best_model_v2.pth
├── face_landmarker.task
│
├── requirements.txt
├── README.md
│
├── confusion_matrix_ll.png
├── confusion_matrix_v2.png
├── roc_curve_ll.png
├── roc_curve_v2.png
│
├── eval_results.py
├── eval_results1.py


⚠️ Notes

- Ensure webcam access is enabled
- Works best under moderate lighting
- Extremely dark or overexposed conditions may affect detection

---

📌 Author

Naman Saxena
Nilesh Sahu
