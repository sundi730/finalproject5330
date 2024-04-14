import cv2
import pygame
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
import random

#the puzzle game!!!!
def calculate_angle(hand_landmarks):
    wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x,
                      hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])
    middle_fingertip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x,
                                 hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y])
    vector = middle_fingertip - wrist
    angle = np.arctan2(vector[1], vector[0]) * 180 / np.pi
    return angle

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

def gesture_prediction(frame, hand_landmarks, model):
    # Extract landmarks and preprocess them directly within this function
    preprocessed_landmarks = process_landmarks(frame, hand_landmarks.landmark)
    prediction = model.predict(np.expand_dims(preprocessed_landmarks, axis=0))
    predicted_class = np.argmax(prediction)
    max_probability = np.max(prediction)

    # Check if the predicted class is in class_map
    if predicted_class in class_map.values():
        return predicted_class, max_probability
    else:
        return None, None  # Return None for invalid predictions




class Tile(pygame.sprite.Sprite):
    def __init__(self, image, rect, position, angle):
        super().__init__()
        self.image = image
        self.rect = rect
        self.selected = False  # Initial state is unselected
        self.position = position
        self.angle = 0  # Store the initial angle
        self.rotate(angle)  # Rotate the image initially
        self.snapped = False

    def rotate(self, angle_change):
        # Add angle_change to the current angle
        self.angle += angle_change
        # Ensure the angle stays within the range of 0 to 359 degrees
        self.angle %= 360

        # Rotate the image
        self.image = pygame.transform.rotate(self.image, angle_change)
        # Update the rectangle area
        self.rect = self.image.get_rect(center=self.rect.center)

    def isSnapped(self):
        self.snapped = True
        self.selected = False  # Once snapped, the tile cannot be selected
    def select(self):
        if not self.snapped:
            self.selected = True

    def deselect(self):
        # Deselect the tile
        self.selected = False

    def draw(self, screen):

        if self.snapped:
            pygame.draw.rect(screen, (0, 255, 0), self.rect, 4)
        # Draw a rectangle around the tile
        elif self.selected:
            pygame.draw.rect(screen, (0, 0, 255), self.rect, 2)
        else:
            pygame.draw.rect(screen, (255, 255, 0), self.rect, 2)

    def set_center(self, new_center_x, new_center_y):
        # Set the center of the tile to new coordinates
        self.rect.center = (new_center_x, new_center_y)



# Load the trained model
model = load_model('test_classifier_v7.hdf5')
mp_hands = mp.solutions.hands
# Initialize the camera


hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
# Define the class map for the new gestures
class_map = {'Pointing': 0, 'Cancel': 1, 'Rotating': 2, 'Move': 3}

pygame.init()
# 导入图片
image_path = "puzzle_image.jpg"  # 更换为你的图片路径
image = pygame.image.load(image_path)
image = pygame.transform.scale(image, (270, 270))  # 根据需要调整尺寸

image.set_alpha(200)

background_path = "processed_image.jpg"
background_img = pygame.image.load(background_path)
background_img = pygame.transform.scale(background_img, (270, 270))
background_img.set_alpha(100)





# 分割九个方块 从左到右 从上到下
tile_size = 90
tiles = []
for y in range(3):
    for x in range(3):
        rect = pygame.Rect(x * tile_size, y * tile_size, tile_size, tile_size)
        sub_image = image.subsurface(rect)
        tiles.append(sub_image)

cap = cv2.VideoCapture(0)
# Set the camera resolution to the highest supported
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 假定一个初始宽高比
aspect_ratio = 16 / 9

# 设置窗口的初始大小
window_size = (1280, 720)
window = pygame.display.set_mode(window_size, pygame.RESIZABLE)
active_area_height = window_size[1] * 2 / 3  # 只在屏幕上方2/3区域活动


# Calculate the position to blit the image so that its center aligns with the window center
x_position = (window_size[0] / 2) - (270 / 2)
y_position = (window_size[1] / 2) - (270 / 2)
image_center_position = (x_position, y_position)


# 创建限制区域
active_rect = pygame.Rect(50, 0, window_size[0]-50, active_area_height)
tile_to_select = None
rotation_started = False
# Assuming these are defined at a broader scope

tile_objects = []
for i, tile_image in enumerate(tiles):
    x = random.randint(50, window_size[0] - 140)  # Adjust for margins
    y = random.randint(50, int(active_area_height) - 140)
    angle = random.choice([0, 90, 180, 270])
    rect = pygame.Rect(x, y, tile_size, tile_size)
    tile_obj = Tile(tile_image, rect, (x, y), angle)
    tile_objects.append(tile_obj)

# 图片中心
image_center = (window_size[0] // 2, window_size[1] // 2)
# 目标位置和目标角度
target_positions = []
for i in range(9):
    x = image_center[0] + ((i % 3) - 1) * 90
    y = image_center[1] + ((i // 3) - 1) * 90
    target_positions.append((x, y))

target_angle = 0

def draw_target_positions(window, target_positions):
    for pos in target_positions:
        pygame.draw.rect(window, (255, 0, 0), (pos[0] - 45, pos[1] - 45, 90, 90), 2)  # Draw red rectangles

def check_and_snap(tile_position, tile_angle, tile_index):
    target_x, target_y = target_positions[tile_index]

    # Calculate the center of the tile
    tile_center_x = tile_position[0] + tile_size // 2
    tile_center_y = tile_position[1] + tile_size // 2

    dx = tile_center_x - target_x
    dy = tile_center_y - target_y
    distance = np.sqrt(dx ** 2 + dy ** 2)
    print(f"Distance: {distance}, Angle: {tile_angle}, Target Angle: {target_angle}， target position {target_x, target_y},"
          f"tile position{tile_center_x, tile_center_y}")

    if distance <= position_tolerance and tile_angle == target_angle:
        snapped_positions[tile_index] = True  # Update the snapped_positions list
        return (target_x, target_y), True
    return tile_position, False



position_tolerance = 75
snapped_positions = [False] * 9

last_angle = None
sequence_progress = 0

running = True
while running:
    ret, frame = cap.read()
    if not ret:
        break
    results = hands.process(frame)
    # 进行手势识别和操作
    gesture_label = "No Gesture Recognized"
    max_probability = 0
    rotation_indicator = ""
    current_angle = 0
      # Flip the frame horizontally
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            predicted_class, max_probability = gesture_prediction(frame,hand_landmarks, model)
            gesture_label = next((label for label, cls in class_map.items() if cls == predicted_class), "Unknown")
            # Assuming frame's shape and screen's dimensions are the same or properly scaled

            if gesture_label == "Pointing":
                # Get index finger tip position in pixel coordinates
                sequence_progress = 0  # Reset the sequence
                last_angle = None
                current_width, current_height = window.get_size()
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                index_x = current_width - int(index_tip.x * current_width)  # Adjust for mirroring
                index_y = int(index_tip.y * current_height)
                # Corrected: Use MIDDLE_FINGER_TIP landmark for middle_tip
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                middle_x = current_width - int(middle_tip.x * current_width)  # Adjust for mirroring
                middle_y = int(middle_tip.y * current_height)
                if tile_to_select is None:
                    for obj in tile_objects:
                        if obj.rect.collidepoint(index_x, index_y) and obj.rect.collidepoint(middle_x, middle_y):
                            if obj.snapped == False:
                                obj.select()
                                tile_to_select = obj
                                print(tile_to_select.selected)
                                break

            elif gesture_label == "Cancel":
                sequence_progress = 0  # Reset the sequence
                last_angle = None
                if tile_to_select is not None:
                    tile_to_select.deselect()
                    print(tile_to_select.selected)
                    tile_to_select = None

            elif gesture_label == "Move":
                sequence_progress = 0  # Reset the sequence
                last_angle = None
                if tile_to_select is not None:
                    # Get the current finger position
                    current_width, current_height = window.get_size()
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    index_x = current_width - int(index_tip.x * current_width)
                    index_y = int(index_tip.y * current_height)
                    if tile_to_select.rect.collidepoint(index_x, index_y):
                        # Check if the new position is within the active area
                        if active_rect.collidepoint(index_x, index_y):
                    # Move the selected tile based on the finger movement
                            tile_to_select.set_center(index_x, index_y)
                            snapped_position, snapped = check_and_snap(tile_to_select.rect.center, tile_to_select.angle,
                                                                       tile_objects.index(tile_to_select))
                            if snapped:
                                # 如果满足吸附条件，更新拼图块的位置
                                tile_to_select.set_center(*snapped_position)
                                tile_to_select.isSnapped()
                                tile_to_select = None

            elif gesture_label == "Rotating":
                if tile_to_select is not None:
                    current_angle = calculate_angle(hand_landmarks)
                    if last_angle is None:
                        last_angle = current_angle

                        # Define angles for open hand and palm sideways
                    OPEN_HAND_ANGLE = -90  # Adjust based on your observation
                    LEFT_SIDEWAYS = -70  # Adjust based on your observation
                    RIGHT_SIDEWAYS = -120
                    # Check sequence progress
                    if sequence_progress == 0 and np.abs(current_angle - OPEN_HAND_ANGLE) < 10:
                        sequence_progress = 1
                        last_angle = current_angle  # Save the last angle when hand is open

                    elif sequence_progress == 1:
                        if np.abs(current_angle - LEFT_SIDEWAYS) < 10:
                            direction = 'left'
                            sequence_progress = 2
                            last_angle = current_angle  # Save the last angle at left sideways position
                        elif np.abs(current_angle - RIGHT_SIDEWAYS) < 10:
                            direction = 'right'
                            sequence_progress = 2
                            last_angle = current_angle  # Save the last angle at right sideways position

                    elif sequence_progress == 2 and np.abs(current_angle - OPEN_HAND_ANGLE) < 10:
                        if direction == 'left':
                            tile_to_select.rotate(90)  # Rotate left
                        elif direction == 'right':
                            tile_to_select.rotate(-90)  # Rotate right
                        print("Sequence Completed")
                        sequence_progress = 0  # Reset the sequence
                        last_angle = None



                        # 处理窗口事件
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.VIDEORESIZE:
            # 捕捉窗口大小调整事件
            # 确保窗口按照原始宽高比调整大小
            width, height = event.size  # 获取事件中的新大小

            # 计算新的宽高以保持宽高比
            new_height = int(width / aspect_ratio)
            if new_height <= height:
                new_size = (width, new_height)
            else:
                new_width = int(height * aspect_ratio)
                new_size = (new_width, height)

            window = pygame.display.set_mode(new_size, pygame.RESIZABLE)

    # 将OpenCV的BGR图像转换为RGB图像

    frame = np.rot90(frame)

    # 将numpy数组图像转换为pygame.Surface对象
    frame_surface = pygame.surfarray.make_surface(frame)

    # 调整surface大小以适应窗口
    frame_surface = pygame.transform.scale(frame_surface, window.get_size())

    # 绘制图像到窗口上
    window.blit(frame_surface, (0, 0))

    # # 在摄像头画面上方绘制目标区域的3x3方格

    window.blit(background_img, image_center_position)
    draw_target_positions(window, target_positions)

    # Now, draw the circle at the index finger position
    if gesture_label == "Pointing":
        pygame.draw.circle(window, (255, 0, 0), (index_x, index_y), 5)
        pygame.draw.circle(window, (255, 0, 0), (middle_x, middle_y), 5)

    # 在窗口上绘制方块对象
    for tile_obj in tile_objects:
        window.blit(tile_obj.image, tile_obj.rect)
        tile_obj.draw(window)


    # 在右上角显示预测结果和概率
    font = pygame.font.SysFont(None, 24)  # 选择字体和字号
    text_surface = font.render("Predicted Gesture: " + gesture_label, True, (255, 255, 255))  # 渲染文本为图像表面
    window.blit(text_surface, (1000, 0))  # 绘制文本到右上角位置
    text_surface = font.render("Probability: " + str(max_probability), True, (255, 255, 255))
    window.blit(text_surface, (1000, 15))
    text_surface = font.render("current_angle: " + str(current_angle), True, (255, 255, 255))
    window.blit(text_surface, (1000, 30))
    text_surface = font.render("rotation stage: " + str(sequence_progress), True, (255, 255, 255))
    window.blit(text_surface, (1000, 45))



    game_completed = all(snapped_positions)
    if game_completed:
        font = pygame.font.SysFont(None, 55)
        completion_message = font.render('Puzzle Completed!', True, (0, 255, 0))
        window.blit(completion_message, (window_size[0] // 2 - completion_message.get_width() // 2,
                                         window_size[1] // 2 - completion_message.get_height() // 2))

    pygame.display.flip()

cap.release()
pygame.quit()
