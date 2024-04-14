import cv2
import numpy as np

#process background img!!!!
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 150, 170)  # These thresholds can be tuned

    # Invert the edges image: edges are white, background is black
    edges_inv = cv2.bitwise_not(edges)

    # Create a white background image
    white_background = np.ones_like(edges_inv) * 255

    # Apply the edges on the white background
    output = cv2.bitwise_and(white_background, edges_inv)

    # # Display the result
    # cv2.imshow('Edges on White Background', output)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the result to file
    save_path = 'processed_image.jpg'  # Specify the file path and name
    cv2.imwrite(save_path, output)
    print(f"Image saved to {save_path}")

# Replace 'puzzle_image.jpg' with your image file path
process_image('puzzle_image.jpg')
