import cv2
import numpy as np
import mediapipe as mp
import csv

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the hands module
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Function to normalize hand landmarks
def normalize_landmarks(landmarks):
    # Convert landmarks to array
    landmarks_arr = np.array(landmarks)
    # Reshape landmarks to (21, 2) to separate x and y coordinates
    landmarks_arr = landmarks_arr.reshape((21, 2))
    # Calculate relative coordinates
    relative_landmarks = landmarks_arr - landmarks_arr[0]
    # Flatten the landmarks back to 1D array
    relative_landmarks = relative_landmarks.flatten()
    # Find the maximum absolute value for normalization
    max_abs_val = np.max(np.abs(relative_landmarks))
    # Normalize the relative coordinates
    normalized_landmarks = relative_landmarks / max_abs_val
    return normalized_landmarks

# Function to save hand landmarks and corresponding label to CSV file
def save_to_csv(data, label):
    # Map label to class number
    class_map = {'select': 0, 'cancel': 1, 'rotate': 2, 'move': 3}
    class_number = class_map[label]

    # Normalize landmarks
    normalized_landmarks = normalize_landmarks(data)

    # Save data and class number to CSV file
    with open('hand_landmarks.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([class_number] + normalized_landmarks.tolist())

# Capture video from the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert landmarks to list of [x, y] coordinates
            landmark_coords = []
            for landmark in hand_landmarks.landmark:
                landmark_x = landmark.x
                landmark_y = landmark.y
                landmark_coords.extend([landmark_x, landmark_y])

            # Collect and save data based on key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('0'):  # Gesture for 'select'
                cv2.putText(frame, "Recording landmark for select gesture...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                save_to_csv(np.array(landmark_coords), 'select')
            elif key == ord('1'):  # Gesture for 'cancel'
                cv2.putText(frame, "Recording landmark for cancel gesture...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                save_to_csv(np.array(landmark_coords), 'cancel')
            elif key == ord('2'):  # Gesture for 'rotate'
                cv2.putText(frame, "Recording landmark for rotate gesture...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                save_to_csv(np.array(landmark_coords), 'rotate')
            elif key == ord('3'):  # Gesture for 'move'
                cv2.putText(frame, "Recording landmark for move gesture...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                save_to_csv(np.array(landmark_coords), 'move')

    cv2.imshow('Hand Gestures', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
