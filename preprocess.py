import cv2
import numpy as np
import os
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path):
    """Convert image to grayscale, apply Gaussian blur, and Canny edge detection."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 200)
    return edges

def augment_image(image):
    """Apply random augmentation like flipping, rotation, and noise addition."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    image = np.expand_dims(image, axis=0)  # Expand dimensions for batch processing
    return datagen.flow(image, batch_size=1)[0].astype('uint8')[0]

if __name__ == "__main__":
    sample_img = "dataset/sample_scan.jpg"
    processed = preprocess_image(sample_img)
    augmented = augment_image(processed)
    cv2.imwrite("dataset/processed_scan.jpg", processed)
    cv2.imwrite("dataset/augmented_scan.jpg", augmented)
