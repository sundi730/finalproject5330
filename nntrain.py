import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard, LambdaCallback
import seaborn as sns
from datetime import datetime
import io

# Load data from CSV file excluding the header row
dataset = 'hand_landmarks_v7.csv'
model_save_path = 'test_classifier_v7.hdf5'
data = pd.read_csv(dataset, skiprows=1)

# Split dataset into features (X) and labels (y)
X_dataset = data.values[:, 1:].astype('float32')
y_dataset = data.values[:, 0].astype('int32')

NUM_CLASSES = 4

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((42,)),  # Adjusted to match 21 * 2
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and returns it."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)  # Adds a batch dimension
    return image

# Define class names as a list
class_names = ['select', 'cancel', 'rotate', 'move']
def log_confusion_matrix(epoch, logs):
    test_pred_raw = model.predict(X_test)
    test_pred = np.argmax(test_pred_raw, axis=1)
    cm = confusion_matrix(y_test, test_pred)
    figure = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    cm_image = plot_to_image(figure)

    with file_writer.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")  # Log directory
file_writer = tf.summary.create_file_writer(log_dir + '/cm')
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
cm_callback = LambdaCallback(on_epoch_end=log_confusion_matrix)

# Fit the model with both TensorBoard and the custom confusion matrix callbacks
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=36,
    validation_data=(X_test, y_test),
    callbacks=[tensorboard_callback, cm_callback]
)

model.save(model_save_path)

# Start TensorBoard in your terminal or command prompt:
# tensorboard --logdir=logs/fit
