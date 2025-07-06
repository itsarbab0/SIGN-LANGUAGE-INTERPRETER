# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf

# try:
#     model = tf.keras.models.load_model('best_model.h5')  
#     print("Loaded best model checkpoint")
# except:
#     model = tf.keras.models.load_model('deep_model.h5') 
#     print("Loaded regular model")

# with open('label_encoder.pkl', 'rb') as f:
#     label_encoder = pickle.load(f)

# with open('deep_model_info.pkl', 'rb') as f:
#     model_info = pickle.load(f)

# # Extract the expected feature count from the model
# expected_features = 42  # This is what your error indicates the model expects

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# cap = cv2.VideoCapture(0)
# def get_alphabet_label(label):
#     return chr(65 + int(label))


# while True:
#     data_aux = []
#     x_ = []
#     y_ = []

#     ret, frame = cap.read()
#     if not ret:
#         continue

#     H, W, _ = frame.shape

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     results = hands.process(frame_rgb)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())

#         # Only process the first hand detected
#         hand_landmarks = results.multi_hand_landmarks[0]
#         for i in range(len(hand_landmarks.landmark)):
#             x = hand_landmarks.landmark[i].x
#             y = hand_landmarks.landmark[i].y
#             x_.append(x)
#             y_.append(y)

#         for i in range(len(hand_landmarks.landmark)):
#             x = hand_landmarks.landmark[i].x
#             y = hand_landmarks.landmark[i].y
#             data_aux.append(x - min(x_))
#             data_aux.append(y - min(y_))

#         data_aux = np.asarray(data_aux)
        
#         if len(data_aux) > 0:
#             # Ensure exactly the expected number of features
#             if len(data_aux) > expected_features:
#                 print(f"Warning: Too many features ({len(data_aux)}), trimming to {expected_features}")
#                 data_aux = data_aux[:expected_features]
#             elif len(data_aux) < expected_features:
#                 print(f"Warning: Too few features ({len(data_aux)}), padding to {expected_features}")
#                 padding = np.zeros(expected_features - len(data_aux))
#                 data_aux = np.concatenate([data_aux, padding])
                
#             data_aux = data_aux / np.max(data_aux)
            
#             data_aux = data_aux.reshape(1, expected_features)
            
#             try:
#                 prediction = model.predict(data_aux, verbose=0)
#                 predicted_class = np.argmax(prediction[0])
                
#                 if predicted_class < len(label_encoder.classes_):
#                     predicted_letter = get_alphabet_label(predicted_class)
#                     confidence = prediction[0][predicted_class] * 100
                    
#                     x1 = int(min(x_) * W) - 10
#                     y1 = int(min(y_) * H) - 10
#                     x2 = int(max(x_) * W) - 10
#                     y2 = int(max(y_) * H) - 10
                    
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
#                     cv2.putText(frame, f"{predicted_letter} ({confidence:.1f}%)", 
#                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
#             except Exception as e:
#                 print(f"Prediction error: {e}")
#                 cv2.putText(frame, "Prediction error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow('Sign Language Recognition', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

try:
    model = tf.keras.models.load_model('best_model.h5')  
    print("Loaded best model checkpoint")
except:
    model = tf.keras.models.load_model('deep_model.h5') 
    print("Loaded regular model")

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

with open('deep_model_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

# Extract the expected feature count from the model
expected_features = 42  # This is what your error indicates the model expects

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)
def get_alphabet_label(label):
    return chr(65 + int(label))

# For tracking prediction changes
last_prediction = None
prediction_count = 0
confidence_threshold = 60  # Only show high confidence predictions in terminal

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Only process the first hand detected
        hand_landmarks = results.multi_hand_landmarks[0]
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
        
        if len(data_aux) > 0:
            # Ensure exactly the expected number of features
            if len(data_aux) > expected_features:
                print(f"Warning: Too many features ({len(data_aux)}), trimming to {expected_features}")
                data_aux = data_aux[:expected_features]
            elif len(data_aux) < expected_features:
                print(f"Warning: Too few features ({len(data_aux)}), padding to {expected_features}")
                padding = np.zeros(expected_features - len(data_aux))
                data_aux = np.concatenate([data_aux, padding])
                
            data_aux = data_aux / np.max(data_aux)
            
            data_aux = data_aux.reshape(1, expected_features) 
            
            try:
                prediction = model.predict(data_aux, verbose=0)
                predicted_class = np.argmax(prediction[0])
                
                if predicted_class < len(label_encoder.classes_):
                    predicted_letter = get_alphabet_label(predicted_class)
                    confidence = prediction[0][predicted_class] * 100
                    
                    # Display on frame
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) - 10
                    y2 = int(max(y_) * H) - 10
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                    cv2.putText(frame, f"{predicted_letter} ({confidence:.1f}%)", 
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    
                    # Terminal output for predictions
                    # Only print prediction if it's different from the last one
                    # or has high confidence
                    if predicted_letter != last_prediction or confidence > confidence_threshold:
                        if predicted_letter != last_prediction:
                            prediction_count = 1
                            last_prediction = predicted_letter
                        else:
                            prediction_count += 1
                        
                        # Print prediction with confidence
                        if confidence > confidence_threshold:
                            print(f"Detected: {predicted_letter} (Confidence: {confidence:.1f}%, Count: {prediction_count})")
                            
                        # Show top 3 predictions if confidence is high enough
                        if confidence > 80:
                            # Get top 3 predictions
                            top_indices = np.argsort(prediction[0])[-3:][::-1]
                            print("Top predictions:")
                            for idx in top_indices:
                                pred_letter = get_alphabet_label(idx)
                                pred_conf = prediction[0][idx] * 100
                                print(f"  {pred_letter}: {pred_conf:.1f}%")
                            print("-" * 30)
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, "Prediction error", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Sign Language Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()