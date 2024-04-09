import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp

# Load the trained model
model = load_model('test_classifier.hdf5')

cancel_threshold = 0.75
# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

# Initialize the camera
cap = cv2.VideoCapture(0)

rotation_speed_factor = 0.1  # Controls the speed of rotation
total_rotation_angle = 0

# Define the class map for the new gestures
class_map = {'Pointing': 0, 'Cancel': 1, 'Rotating': 2, 'Move': 3}

# Load and prepare the image
image_path = 'img.jpg'  # Update this path
input_image = cv2.imread(image_path)
if input_image is None:
    print("Error: Image not found.")
    exit(1)
input_image = cv2.resize(input_image, (200, 200))  # Resize image


# Load the original image
original_image = cv2.imread(image_path)
original_image = cv2.resize(original_image, (200, 200))
# Image parameters
image_x, image_y = 200, 200  # Initial position of the image
image_selected = False

# Rotation parameters
rotation_started = False
initial_vector = None
initial_angle = 0

highlight_color = (0, 255, 0)  # Green color for highlighting

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = np.array([[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in hand_landmarks.landmark])
            preprocessed_landmarks = (landmarks.flatten() - landmarks.flatten()[0]) / max(np.abs(landmarks.flatten()))
            prediction = model.predict(np.expand_dims(preprocessed_landmarks, axis=0))
            predicted_class = np.argmax(prediction)
            max_probability = np.max(prediction)
            gesture_label = next((label for label, cls in class_map.items() if cls == predicted_class), "Unknown")
            # Update frame with gesture label, probability, and rotation angle
            cv2.putText(frame, f'Gesture: {gesture_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Probability: {max_probability:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            if predicted_class == class_map['Pointing']:
                # Logic to select the image
                if image_x <= landmarks[8][0] <= image_x + 200 and image_y <= landmarks[8][1] <= image_y + 200:
                    image_selected = True

            elif predicted_class == class_map['Cancel'] and max_probability > cancel_threshold:

                image_selected = False
                rotation_started = False

            elif predicted_class == class_map['Move'] and image_selected:
                image_x = int(landmarks[8][0] - 100)
                image_y = int(landmarks[8][1] - 100)

            elif predicted_class == class_map['Rotating'] and image_selected:
                if not rotation_started:
                    initial_vector = np.array([landmarks[12][0] - landmarks[0][0], landmarks[12][1] - landmarks[0][1]])
                    initial_angle = np.degrees(np.arctan2(initial_vector[1], initial_vector[0]))
                    rotation_started = True
                else:
                    # Compute the current angle
                    current_vector = np.array([landmarks[12][0] - landmarks[0][0], landmarks[12][1] - landmarks[0][1]])
                    current_angle = np.degrees(np.arctan2(current_vector[1], current_vector[0]))
                    rotation_angle_change = current_angle - initial_angle


                    # Update the total rotation angle slowly based on the rotation_speed_factor
                    total_rotation_angle += rotation_angle_change * rotation_speed_factor

                    # Display rotation indicator
                    rotation_indicator = f"Total Rotation Angle: {total_rotation_angle:.2f} degrees"
                    cv2.putText(frame, rotation_indicator, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    # Always rotate from the original image to minimize blurring
                    rotation_matrix = cv2.getRotationMatrix2D((100, 100), -total_rotation_angle, 1.0)
                    input_image = cv2.warpAffine(original_image, rotation_matrix, (200, 200))

    # Place the input image on the frame (adjusted for movement and rotation)
    frame[image_y:image_y+200, image_x:image_x+200] = input_image

    # Highlight the image if selected
    if image_selected:
        cv2.rectangle(frame, (image_x, image_y), (image_x + 200, image_y + 200), highlight_color, 2)

    # Show the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
