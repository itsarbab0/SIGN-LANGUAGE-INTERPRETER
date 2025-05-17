# Sign Language Interpreter

A web application for real-time sign language detection and learning using AI. This application detects hand gestures for American Sign Language (ASL) letters A through I.

## Features

- **Real-time Sign Language Recognition**: Use your webcam to detect and interpret sign language gestures
- **Learning Resources**: Learn ASL letters A through I with visual guides and tips
- **Interactive Practice**: Practice signing and get real-time feedback on your gestures
- **AI-Powered**: Utilizes deep learning and computer vision for accurate gesture recognition

## Tech Stack

- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript with Bootstrap 5
- **AI/ML**: TensorFlow, MediaPipe for hand tracking
- **Computer Vision**: OpenCV

## Prerequisites

- Python 3.8 or higher
- Webcam for real-time detection
- Modern web browser (Chrome, Firefox, Edge recommended)

## Setup Instructions

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv venv
   ```
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install required packages:
   ```
   pip install -r requirements.txt
   ```
5. Copy your trained model files to the models folder:
   - deep_model.h5
   - label_encoder.pkl
   - deep_model_info.pkl
6. Run the application:
   ```
   python app.py
   ```
7. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Deployment

This application can be deployed to platforms like Heroku, Vercel, or any other hosting service that supports Python applications.

## Project Structure

```
sign-language-app/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── models/             # Contains the trained AI models
├── static/             # CSS, JavaScript and static assets
│   └── css/
│       └── style.css   # Application styles
└── templates/          # HTML templates
    ├── index.html      # Home page
    ├── learning.html   # Learning resources page
    └── practice.html   # Practice with webcam page
```

## Credits

- ASL reference images from [HandSpeak](https://www.handspeak.com/)
- Bootstrap for UI components
- TensorFlow and MediaPipe for AI capabilities

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

Developed for an AI course project, focused on practical applications of machine learning in everyday life. 