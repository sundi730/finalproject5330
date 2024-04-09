import pygame
import cv2
import numpy as np
import random
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

# Define the class map for the new gestures
class_map = {'Pointing': 0, 'Cancel': 1, 'Rotating': 2, 'Move': 3}


pygame.init()
# 在pygame初始化后创建字体对象
myfont = pygame.font.SysFont('Arial', 30)
# 摄像头初始化
cap = cv2.VideoCapture(0)

# Pygame窗口设置
window_size = (1200, 600)
screen = pygame.display.set_mode(window_size)
pygame.display.set_caption("Puzzle Game")

# 导入图片
image_path = "puzzle_image.jpg"  # 更换为你的图片路径
image = pygame.image.load(image_path)
image = pygame.transform.scale(image, (270, 270))  # 根据需要调整尺寸

# 图片中心
image_center = (window_size[0] // 2, window_size[1] // 2)

# 目标位置和目标角度
target_positions = {}
for i in range(9):
    x = image_center[0] + ((i % 3) - 1) * 90
    y = image_center[1] + ((i // 3) - 1) * 90
    target_positions[i] = (x, y)

target_angle = 0

# 随机化位置和旋转角度
random_positions = []
random_angles = []
for _ in range(9):
    x = random.randint(50, window_size[0] - 140)  # 留出边缘空间
    y = random.randint(50, window_size[1] - 140)
    random_positions.append((x, y))
    random_angles.append(random.choice([0, 90, 180, 270]))

snapped_positions = [False] * 9  # 跟踪每个拼图块是否已经被放置到正确的位置

# 拼图块拖拽逻辑变量
dragging = False
dragged_tile_index = None
mouse_offset = (0, 0)

# 检查并吸附函数
def check_and_snap(tile_position, tile_angle):
    for i, (target_x, target_y) in target_positions.items():
        dx = tile_position[0] - target_x
        dy = tile_position[1] - target_y
        distance = np.sqrt(dx ** 2 + dy ** 2)
        if distance <= position_tolerance and tile_angle == target_angle:
            return (target_x, target_y), True
    return tile_position, False

position_tolerance = 50

# 游戏主循环
running = True
while running:
    ret, frame = cap.read()
    if not ret:
        continue

    # 将捕获的图像从BGR转换为RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    # 将numpy数组图像转换为pygame.Surface对象
    frame_surface = pygame.surfarray.make_surface(frame)

    # 旋转后，宽度和高度互换
    rotated_width = frame.shape[0]  # 原始高度成为新的宽度
    rotated_height = frame.shape[1]  # 原始宽度成为新的高度

    # 将numpy数组图像转换为pygame.Surface对象
    frame_surface = pygame.surfarray.make_surface(frame)

    # 计算图像缩放比例（使用旋转后的尺寸）
    scale_width = window_size[0] / rotated_width
    scale_height = window_size[1] / rotated_height
    scale = min(scale_width, scale_height)

    # 计算新的图像尺寸（使用旋转后的尺寸）
    new_width = int(rotated_width * scale)
    new_height = int(rotated_height * scale)

    # 缩放图像
    frame_surface = pygame.transform.scale(frame_surface, (new_width, new_height))



    # 将调整后的图像绘制到窗口的中心
    screen.blit(frame_surface, ((window_size[0] - new_width) // 2, (window_size[1] - new_height) // 2))

    gesture_label = "No Gesture Recognized"
    max_probability = 0
    rotation_indicator = ""
    results = hands.process(frame_surface)


    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False


        # 在MOUSEBUTTONDOWN事件中反向遍历方块列表

        elif event.type == pygame.MOUSEBUTTONDOWN and not dragging:

            mouse_pos = pygame.mouse.get_pos()

            # 使用reversed函数反向遍历列表

            for i, pos in enumerate(reversed(random_positions)):

                # 调整索引以反映正确的方块索引

                actual_index = len(random_positions) - 1 - i

                # 计算矩形左上角的位置，而不是使用中心点

                tile_rect = pygame.Rect(pos[0] - 45, pos[1] - 45, 90, 90)

                if tile_rect.collidepoint(mouse_pos):
                    dragging = True

                    # 使用actual_index而不是i，以确保正确的方块被选中

                    dragged_tile_index = actual_index

                    mouse_offset = (mouse_pos[0] - pos[0], mouse_pos[1] - pos[1])

                    break  # 一旦找到被点击的方块，立即停止循环


        elif event.type == pygame.MOUSEBUTTONUP and dragging:
            new_pos, snapped = check_and_snap(random_positions[dragged_tile_index], random_angles[dragged_tile_index])
            random_positions[dragged_tile_index] = new_pos
            snapped_positions[dragged_tile_index] = snapped
            dragging = False
            dragged_tile_index = None

        elif event.type == pygame.MOUSEMOTION and dragging:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            random_positions[dragged_tile_index] = (mouse_x - mouse_offset[0], mouse_y - mouse_offset[1])

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and dragged_tile_index is not None:
                random_angles[dragged_tile_index] = (random_angles[dragged_tile_index] + 90) % 360

    # 绘制背景和拼图块
    #screen.fill((0, 0, 0))  # 清屏
    for i in range(9):
        tile = pygame.transform.rotate(image.subsurface(((i % 3) * 90, (i // 3) * 90, 90, 90)), random_angles[i])
        tile_rect = tile.get_rect(center=random_positions[i])
        screen.blit(tile, tile_rect.topleft)

        # 如果当前拼图块被拖拽，用蓝色高亮
        if i == dragged_tile_index:
            pygame.draw.rect(screen, (0, 0, 255), tile_rect, 3)

        # 绘制数字
        text_surface = myfont.render(str(i + 1), True, (255, 255, 255))
        screen.blit(text_surface, (tile_rect.left + 5, tile_rect.top + 5))

        if snapped_positions[i]:
            pygame.draw.rect(screen, (0, 255, 0), tile_rect, 3)  # 用绿色边框标记已正确放置的拼图块

    # 在摄像头画面上方绘制目标区域的3x3方格
    for idx, pos in target_positions.items():
        pygame.draw.rect(screen, (255, 0, 0), (pos[0] - 45, pos[1] - 45, 90, 90), 2)  # 绘制红色提示框

    game_completed = all(snapped_positions)
    if game_completed:
        font = pygame.font.SysFont(None, 55)
        completion_message = font.render('Puzzle Completed!', True, (0, 255, 0))
        screen.blit(completion_message, (window_size[0] // 2 - completion_message.get_width() // 2,
                                         window_size[1] // 2 - completion_message.get_height() // 2))

    pygame.display.flip()  # 更新屏幕显示

pygame.quit()
