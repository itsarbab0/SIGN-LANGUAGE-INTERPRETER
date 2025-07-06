import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import pickle
from PIL import Image, ImageDraw, ImageFont
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd

# Fixed MediaPipe import
try:
    import mediapipe as mp
    # Modern MediaPipe import structure
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_AVAILABLE = True
    st.success("✅ MediaPipe loaded successfully!")
except ImportError as e:
    st.error("⚠️ MediaPipe not found. Please install it:")
    st.code("pip install mediapipe", language="bash")
    MEDIAPIPE_AVAILABLE = False
except Exception as e:
    st.error(f"⚠️ MediaPipe import error: {str(e)}")
    st.info("Try reinstalling MediaPipe:")
    st.code("pip uninstall mediapipe && pip install mediapipe", language="bash")
    MEDIAPIPE_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="ASL Alphabet Detection",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 1rem 0;
    }
    .confidence-meter {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Load model and label encoder
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('deep_model.h5')
        return model
    except Exception as e:
        st.error(f"Model file 'deep_model.h5' not found. Error: {str(e)}")
        st.info("Please ensure the model file is in the correct directory.")
        return None

@st.cache_resource
def load_label_encoder():
    try:
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        return label_encoder
    except Exception as e:
        st.error(f"Label encoder file 'label_encoder.pkl' not found. Error: {str(e)}")
        return None

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'correct_predictions' not in st.session_state:
    st.session_state.correct_predictions = 0

model = load_model()
label_encoder = load_label_encoder()

# Check if essential components are loaded
if not MEDIAPIPE_AVAILABLE:
    st.error("❌ MediaPipe is required for hand detection. Please install it and restart the app.")
    st.stop()

if model is None:
    st.error("❌ Model file is required. Please ensure 'deep_model.h5' is available.")
    st.stop()

if label_encoder is None:
    st.error("❌ Label encoder is required. Please ensure 'label_encoder.pkl' is available.")
    st.stop()

EXPECTED_LENGTH = 42

# Header
st.markdown('<h1 class="main-header">🤟 ASL Alphabet Detection</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🎛️ Control Panel")
    
    # Input method selection
    option = st.selectbox(
        '📥 Choose input method:',
        ['Upload Image', 'Use Webcam', 'Practice Mode'],
        help="Select how you want to input your ASL signs"
    )
    
    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    confidence_threshold = st.slider(
        "🎯 Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence required for prediction"
    )
    
    show_landmarks = st.checkbox("👁️ Show Hand Landmarks", value=False)
    
    st.divider()
    
    # Statistics
    st.header("📊 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Predictions", st.session_state.total_predictions)
    with col2:
        accuracy = (st.session_state.correct_predictions / max(st.session_state.total_predictions, 1)) * 100
        st.metric("Accuracy", f"{accuracy:.1f}%")
    
    # Clear history button
    if st.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.session_state.total_predictions = 0
        st.session_state.correct_predictions = 0
        st.success("History cleared!")

# Helper functions
def extract_landmarks(image):
    """Extract hand landmarks from image using MediaPipe"""
    try:
        # Initialize MediaPipe Hands
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        ) as hands:
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            data_aux = []
            x_ = []
            y_ = []
            landmarks_coords = None
            
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                landmarks_coords = []
                
                # Extract coordinates
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)
                    landmarks_coords.append((x, y))
                
                # Normalize coordinates relative to bounding box
                min_x, min_y = min(x_), min(y_)
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    data_aux.append(x - min_x)
                    data_aux.append(y - min_y)
                
                # Ensure we have the expected number of features
                if len(data_aux) == EXPECTED_LENGTH:
                    return np.array(data_aux), landmarks_coords
                else:
                    st.warning(f"Expected {EXPECTED_LENGTH} features, got {len(data_aux)}")
            
            return None, landmarks_coords
            
    except Exception as e:
        st.error(f"Error in landmark extraction: {str(e)}")
        return None, None

def predict_with_confidence(image):
    """Make prediction with confidence score"""
    landmarks, landmarks_coords = extract_landmarks(image)
    if landmarks is None:
        return None, None, None, landmarks_coords
    
    try:
        # Normalize landmarks
        if np.max(landmarks) != 0:
            landmarks = landmarks / np.max(landmarks)
        landmarks = landmarks.reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(landmarks, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = float(prediction[0][predicted_class])
        predicted_letter = chr(65 + predicted_class)  # Convert to letter A-Z
        
        # Get all class probabilities for visualization
        all_probabilities = prediction[0]
        
        return predicted_letter, confidence, all_probabilities, landmarks_coords
        
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        return None, None, None, landmarks_coords

def draw_landmarks_on_image(image, landmarks_coords):
    """Draw hand landmarks on the image"""
    if landmarks_coords is None:
        return image
    
    try:
        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        
        h, w = image.shape[:2]
        
        # Draw landmarks
        for x, y in landmarks_coords:
            pixel_x, pixel_y = int(x * w), int(y * h)
            draw.ellipse([pixel_x-3, pixel_y-3, pixel_x+3, pixel_y+3], fill='red', outline='white')
        
        return np.array(img_pil)
    except Exception as e:
        st.error(f"Error drawing landmarks: {str(e)}")
        return image

def create_confidence_chart(probabilities, predicted_letter):
    """Create a confidence chart showing all letter probabilities"""
    try:
        letters = [chr(65 + i) for i in range(26)]
        
        # Ensure we have enough probabilities
        probs_to_use = probabilities[:26] if len(probabilities) >= 26 else list(probabilities) + [0] * (26 - len(probabilities))
        
        # Create DataFrame
        df = pd.DataFrame({
            'Letter': letters,
            'Confidence': probs_to_use
        })
        
        # Highlight predicted letter
        colors = ['#ff6b6b' if letter == predicted_letter else '#4ecdc4' for letter in df['Letter']]
        
        fig = px.bar(
            df, 
            x='Letter', 
            y='Confidence',
            title='Confidence Scores for All Letters',
            color_discrete_sequence=colors
        )
        
        fig.update_layout(
            xaxis_title="ASL Letters",
            yaxis_title="Confidence Score",
            showlegend=False,
            height=400
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def add_prediction_to_history(predicted_letter, confidence, is_correct=None):
    """Add prediction to session history"""
    st.session_state.prediction_history.append({
        'timestamp': datetime.now(),
        'letter': predicted_letter,
        'confidence': confidence,
        'is_correct': is_correct
    })
    st.session_state.total_predictions += 1
    if is_correct:
        st.session_state.correct_predictions += 1

# Main content area
if option == 'Upload Image':
    st.header("📤 Upload Image Mode")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            '📁 Upload an image of your hand sign',
            type=['jpg', 'jpeg', 'png'],
            help="Upload a clear image showing ASL hand sign"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            
            # Process prediction
            predicted_letter, confidence, all_probs, landmarks_coords = predict_with_confidence(image_np)
            
            # Display image with optional landmarks
            display_image = image_np.copy()
            if show_landmarks and landmarks_coords:
                display_image = draw_landmarks_on_image(display_image, landmarks_coords)
            
            st.image(display_image, caption='Uploaded Image', use_column_width=True)
            
            if predicted_letter and confidence is not None and confidence >= confidence_threshold:
                # Prediction result
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Prediction Result</h2>
                    <h1 style="font-size: 4rem; margin: 1rem 0;">{predicted_letter}</h1>
                    <h3>Confidence: {confidence*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback buttons
                col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 1])
                with col_fb1:
                    if st.button("✅ Correct"):
                        add_prediction_to_history(predicted_letter, confidence, True)
                        st.success("Thanks for the feedback!")
                with col_fb2:
                    if st.button("❌ Incorrect"):
                        add_prediction_to_history(predicted_letter, confidence, False)
                        st.info("Thanks for the feedback!")
                with col_fb3:
                    if st.button("🤷 Not Sure"):
                        add_prediction_to_history(predicted_letter, confidence, None)
                        st.info("Prediction recorded!")
                
            elif predicted_letter and confidence is not None:
                st.warning(f'Low confidence prediction: {predicted_letter} ({confidence*100:.1f}%)')
                st.info(f'Confidence is below threshold ({confidence_threshold*100:.1f}%)')
            else:
                st.error('❌ No hand detected. Please try another image with a clear hand sign.')
    
    with col2:
        if uploaded_file is not None and predicted_letter and all_probs is not None:
            # Confidence visualization
            fig = create_confidence_chart(all_probs, predicted_letter)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

elif option == 'Use Webcam':
    st.header("📹 Webcam Mode")
    
    st.info('📸 Click "Take a picture" to capture and analyze your ASL sign')
    
    camera_image = st.camera_input('Take a picture')
    
    if camera_image is not None:
        image = Image.open(camera_image).convert('RGB')
        image_np = np.array(image)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            predicted_letter, confidence, all_probs, landmarks_coords = predict_with_confidence(image_np)
            
            # Display image with optional landmarks
            display_image = image_np.copy()
            if show_landmarks and landmarks_coords:
                display_image = draw_landmarks_on_image(display_image, landmarks_coords)
            
            st.image(display_image, caption='Captured Image', use_column_width=True)
            
            if predicted_letter and confidence is not None and confidence >= confidence_threshold:
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Prediction Result</h2>
                    <h1 style="font-size: 4rem; margin: 1rem 0;">{predicted_letter}</h1>
                    <h3>Confidence: {confidence*100:.1f}%</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback buttons
                col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 1])
                with col_fb1:
                    if st.button("✅ Correct", key="webcam_correct"):
                        add_prediction_to_history(predicted_letter, confidence, True)
                        st.success("Thanks for the feedback!")
                with col_fb2:
                    if st.button("❌ Incorrect", key="webcam_incorrect"):
                        add_prediction_to_history(predicted_letter, confidence, False)
                        st.info("Thanks for the feedback!")
                with col_fb3:
                    if st.button("🤷 Not Sure", key="webcam_unsure"):
                        add_prediction_to_history(predicted_letter, confidence, None)
                        st.info("Prediction recorded!")
                        
            elif predicted_letter and confidence is not None:
                st.warning(f'Low confidence prediction: {predicted_letter} ({confidence*100:.1f}%)')
            else:
                st.error('❌ No hand detected. Please try again with a clear hand sign.')
        
        with col2:
            if predicted_letter and all_probs is not None:
                fig = create_confidence_chart(all_probs, predicted_letter)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

elif option == 'Practice Mode':
    st.header("🎓 Practice Mode")
    
    # Random letter challenge
    if 'practice_letter' not in st.session_state:
        st.session_state.practice_letter = chr(65 + np.random.randint(0, 26))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Show me the letter:</h2>
            <h1 style="font-size: 6rem; margin: 1rem 0;">{st.session_state.practice_letter}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎲 New Random Letter"):
            st.session_state.practice_letter = chr(65 + np.random.randint(0, 26))
            st.rerun()
    
    with col2:
        camera_image = st.camera_input('Show me your sign!')
        
        if camera_image is not None:
            image = Image.open(camera_image).convert('RGB')
            image_np = np.array(image)
            
            predicted_letter, confidence, all_probs, landmarks_coords = predict_with_confidence(image_np)
            
            if predicted_letter and confidence is not None and confidence >= confidence_threshold:
                is_correct = predicted_letter == st.session_state.practice_letter
                
                if is_correct:
                    st.success(f"🎉 Correct! You signed '{predicted_letter}' with {confidence*100:.1f}% confidence!")
                    add_prediction_to_history(predicted_letter, confidence, True)
                else:
                    st.error(f"❌ You signed '{predicted_letter}' but the target was '{st.session_state.practice_letter}'")
                    add_prediction_to_history(predicted_letter, confidence, False)
                    
                # Show new challenge button
                if st.button("➡️ Next Challenge"):
                    st.session_state.practice_letter = chr(65 + np.random.randint(0, 26))
                    st.rerun()
                    
            elif predicted_letter and confidence is not None:
                st.warning(f'Low confidence: {predicted_letter} ({confidence*100:.1f}%)')
            else:
                st.error('❌ No hand detected. Try again!')

# History section
if st.session_state.prediction_history:
    st.divider()
    st.header("📈 Prediction History")
    
    # Convert history to DataFrame for better display
    history_df = pd.DataFrame(st.session_state.prediction_history)
    history_df['timestamp'] = history_df['timestamp'].dt.strftime('%H:%M:%S')
    history_df['confidence'] = (history_df['confidence'] * 100).round(1)
    
    # Display recent predictions
    st.dataframe(
        history_df[['timestamp', 'letter', 'confidence', 'is_correct']].tail(10),
        use_container_width=True,
        column_config={
            "timestamp": "Time",
            "letter": "Letter",
            "confidence": st.column_config.NumberColumn("Confidence (%)", format="%.1f"),
            "is_correct": "Correct?"
        }
    )
    
    # Performance chart
    if len(history_df) > 1:
        fig_performance = px.line(
            history_df.reset_index(), 
            x='index', 
            y='confidence',
            title='Confidence Over Time',
            labels={'index': 'Prediction Number', 'confidence': 'Confidence (%)'}
        )
        st.plotly_chart(fig_performance, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>🤟 ASL Alphabet Detection System | Built with ❤️ using Streamlit</p>
    <p><small>Tip: For best results, ensure good lighting and clear hand positioning</small></p>
</div>
""", unsafe_allow_html=True)