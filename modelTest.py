import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('test_classifier_v7.hdf5')

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Define the class map for the new gestures
class_map = {'Pointing': 0, 'Cancel': 1, 'Rotating': 2, 'Move': 3}


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    gesture_label = "No Gesture Recognized"
    max_probability = 0
    rotation_indicator = ""


    def process_landmarks(frame, landmarks):
        h, w, _ = frame.shape
        wrist_coords = np.array([landmarks[0].x * w, landmarks[0].y * h])
        relative_coords = []
        for landmark in landmarks:
            x_pixel = landmark.x * w
            y_pixel = landmark.y * h
            # Calculate relative position with respect to the wrist
            relative_x = x_pixel - wrist_coords[0]
            relative_y = y_pixel - wrist_coords[1]
            # Re-normalize
            relative_coords.extend([relative_x / w, relative_y / h])
        return relative_coords

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Preprocess landmarks
            preprocessed_landmarks = process_landmarks(frame, hand_landmarks.landmark)

            # Make prediction
            prediction = model.predict(np.expand_dims(preprocessed_landmarks, axis=0))
            predicted_class = np.argmax(prediction)
            max_probability = np.max(prediction)
            gesture_label = next((label for label, cls in class_map.items() if cls == predicted_class), "Unknown")
            # Update frame with gesture label, probability, and rotation angle
            cv2.putText(frame, f'Gesture: {gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Probability: {max_probability:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)




    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
