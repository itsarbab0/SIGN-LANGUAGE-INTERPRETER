from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import os
import base64
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load the model and required files
model_path = os.path.join('models', 'deep_model.h5')
label_encoder_path = os.path.join('models', 'label_encoder.pkl')
model_info_path = os.path.join('models', 'deep_model_info.pkl')

# Load model
try:
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load label encoder
try:
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    print(f"Loaded label encoder from {label_encoder_path}")
except Exception as e:
    print(f"Error loading label encoder: {e}")
    label_encoder = None

# Load model info
try:
    with open(model_info_path, 'rb') as f:
        model_info = pickle.load(f)
    print(f"Loaded model info from {model_info_path}")
except Exception as e:
    print(f"Error loading model info: {e}")
    model_info = None

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Expected features from the model
expected_features = 42  # This should match your model's input shape

def get_alphabet_label(label):
    """Convert numeric label to alphabet letter"""
    return chr(65 + int(label))

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/learning')
def learning():
    """Render the learning page"""
    return render_template('learning.html')

@app.route('/practice')
def practice():
    """Render the practice page"""
    return render_template('practice.html')

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """Process a frame and return prediction"""
    from flask import request
    
    if request.method == 'POST':
        try:
            # Get the frame data from the request
            data = request.json
            frame_data = data.get('frame')
            
            if not frame_data:
                return jsonify({'error': 'No frame data provided'})
            
            # Decode the base64 frame data
            frame_data = frame_data.split(',')[1]
            frame_bytes = base64.b64decode(frame_data)
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Failed to decode frame'})
            
            # Process the frame with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            result = {
                'detected': False,
                'letter': None,
                'confidence': 0,
                'top_predictions': []
            }
            
            with mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3) as hands:
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    # Process landmarks
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    data_aux = []
                    x_ = []
                    y_ = []
                    
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                    
                    data_aux = np.asarray(data_aux)
                    
                    # Ensure we have the expected number of features
                    if len(data_aux) > expected_features:
                        data_aux = data_aux[:expected_features]
                    elif len(data_aux) < expected_features:
                        padding = np.zeros(expected_features - len(data_aux))
                        data_aux = np.concatenate([data_aux, padding])
                    
                    # Normalize data
                    data_aux = data_aux / np.max(data_aux)
                    
                    # Reshape for model input
                    data_aux = data_aux.reshape(1, expected_features)
                    
                    # Make prediction
                    prediction = model.predict(data_aux, verbose=0)
                    predicted_class = np.argmax(prediction[0])
                    
                    if predicted_class < len(label_encoder.classes_):
                        predicted_letter = get_alphabet_label(predicted_class)
                        confidence = float(prediction[0][predicted_class] * 100)
                        
                        result['detected'] = True
                        result['letter'] = predicted_letter
                        result['confidence'] = confidence
                        
                        # Get top 3 predictions
                        top_indices = np.argsort(prediction[0])[-3:][::-1]
                        top_predictions = []
                        
                        for idx in top_indices:
                            pred_letter = get_alphabet_label(idx)
                            pred_conf = float(prediction[0][idx] * 100)
                            top_predictions.append({
                                'letter': pred_letter,
                                'confidence': pred_conf
                            })
                        
                        result['top_predictions'] = top_predictions
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 
# For Vercel deployment
app.debug = False
