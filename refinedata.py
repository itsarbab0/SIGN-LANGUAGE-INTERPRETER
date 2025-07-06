

import os
import pickle
import mediapipe as mp
import cv2
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

total_images = 0
processed_images = 0
failed_images = 0

print("Starting data processing...")
print(f"Found {len(os.listdir(DATA_DIR))} classes in {DATA_DIR}")

EXPECTED_LENGTH = 42

for dir_ in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, dir_)
    class_images = os.listdir(class_path)
    total_images += len(class_images)
    print(f"\nProcessing class {dir_} ({chr(65 + int(dir_))}) - {len(class_images)} images")
    
    for img_path in os.listdir(class_path):
        data_aux = []
        x_ = []
        y_ = []

        img = cv2.imread(os.path.join(class_path, img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            failed_images += 1
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
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

                if len(data_aux) == EXPECTED_LENGTH:
                    data.append(data_aux)
                    labels.append(dir_)
                    processed_images += 1
                else:
                    print(f"Invalid data length in {img_path}: {len(data_aux)}")
                    failed_images += 1
        else:
            print(f"No hand detected in: {img_path}")
            failed_images += 1

print("\nProcessing Summary:")
print(f"Total images: {total_images}")
print(f"Successfully processed: {processed_images}")
print(f"Failed to process: {failed_images}")

print("\nData verification:")
print(f"Number of data points: {len(data)}")
print(f"Number of labels: {len(labels)}")
if len(data) > 0:
    print(f"Shape of first data point: {len(data[0])}")
    print(f"Unique labels: {np.unique(labels)}")

data = np.array(data)
print(f"\nFinal data shape: {data.shape}")

print("\nSaving data to pickle file...")
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
print("Data saved successfully!")

print("\nVerifying saved pickle file...")
with open('data.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
    print(f"Loaded data shape: {loaded_data['data'].shape}")
    print(f"Loaded labels: {len(loaded_data['labels'])}")
    print("Pickle file verification complete!")