import cv2
import mediapipe as mp
import csv
import numpy as np

#landmark capture official!!!!!

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 打开CSV文件
csv_file = open('hand_landmarks_v2.csv', 'a', newline='')
csv_writer = csv.writer(csv_file)

# 写入CSV表头
header = ['gesture_id']
for i in range(21):
    header.extend([f'x{i}', f'y{i}'])
csv_writer.writerow(header)

def process_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    wrist_coords = np.array([landmarks[0].x * w, landmarks[0].y * h])
    relative_coords = []
    for landmark in landmarks:
        x_pixel = landmark.x * w
        y_pixel = landmark.y * h
        # 计算相对于手腕的位置
        relative_x = x_pixel - wrist_coords[0]
        relative_y = y_pixel - wrist_coords[1]
        # 重新归一化
        relative_coords.extend([relative_x / w, relative_y / h])
    return relative_coords

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmark_list = [landmark for landmark in hand_landmarks.landmark]
            relative_landmarks = process_landmarks(frame, landmark_list)

            # 按键操作保存手势数据
            key = cv2.waitKey(1)
            if key in [ord('0'), ord('1'), ord('2'), ord('3')]:
                gesture_id = key - ord('0')
                csv_writer.writerow([gesture_id] + relative_landmarks)
                print(f"Saved gesture {gesture_id}")

    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()
