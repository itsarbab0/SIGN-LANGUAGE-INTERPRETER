# 🤟 SIGN LANGUAGE INTERPRETER

Live deployment: 👉 [asl-interpreter.streamlit.app](https://asl-interpreter.streamlit.app/)

A real-time American Sign Language (ASL) interpreter built using **MediaPipe**, **TensorFlow**, and **Streamlit**. This is a full-stack **Software Engineering + Machine Learning** project that captures hand gestures from a webcam, classifies them using a trained deep learning model, and translates them into English characters or words.

---

## 🚀 Features

- 🖐️ Real-time hand gesture detection via webcam
- 🧠 Pre-trained deep learning model (`.h5`) for classification
- 🎯 Integrated with **MediaPipe** for accurate hand landmark tracking
- 💻 Web app interface using **Streamlit**
- 🎥 Webcam-based data collection script
- ⚙️ Scripts to train and test custom models
- 📦 Deployed on [Streamlit Cloud](https://asl-interpreter.streamlit.app/)

---

## 🧰 Tech Stack

| Layer             | Technology                          |
|------------------|--------------------------------------|
| Frontend          | Streamlit (Python-based UI framework) |
| Backend / Logic   | Python, OpenCV, MediaPipe, TensorFlow |
| ML Model          | Keras `.h5` model trained on custom ASL data |
| Deployment        | Streamlit Cloud                     |
| Dataset (optional)| Collected via webcam (`collect_img.py`) |

---

## 📂 File Structure

```
├── app.py                # Main Streamlit app
├── collect_img.py        # Script to collect gesture images
├── train_deep.py         # Model training script
├── test_deep.py          # Script to test model accuracy
├── deep_model.h5         # Trained Keras model
├── data.pickle           # Pickled data (optional)
├── label_encoder.pkl     # Label encoder for model predictions
├── deep_model_info.pkl   # Additional model metadata
├── requirements.txt      # Project dependencies
├── runtime.txt           # Python runtime version
├── packages.txt          # OS-level dependencies (optional)
```

---

## 💻 How to Run Locally

### 🧱 Prerequisites
- Python 3.10+
- pip

### 🔧 Installation

```bash
git clone https://github.com/itsarbab0/SIGN-LANGUAGE-INTERPRETER.git
cd SIGN-LANGUAGE-INTERPRETER

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

### ▶️ Run the App

```bash
streamlit run app.py
```

Once launched, allow webcam access and start signing!

---

## 🧪 Train Your Own Model

If you'd like to train your own gesture classifier:

```bash
python collect_img.py    # Collect custom images using webcam
python train_deep.py     # Train the model
python test_deep.py      # Test the trained model
```

Your model will be saved as `deep_model.h5`.

---

## 🌐 Deployment

This app is deployed at:

👉 **[https://asl-interpreter.streamlit.app/](https://asl-interpreter.streamlit.app/)**

It is hosted on [Streamlit Cloud](https://streamlit.io/cloud).

---

## 👤 Author

**Arbab Kareem**  
GitHub: [@itsarbab0](https://github.com/itsarbab0)

---

## 📜 License

This project is for educational purposes. You are free to fork and extend it with proper credit.

---

## 🙌 Acknowledgements

- [Streamlit](https://streamlit.io/)
- [MediaPipe](https://google.github.io/mediapipe/)
- [TensorFlow](https://www.tensorflow.org/)
