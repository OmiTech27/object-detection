import tensorflow as tf
import cv2
import numpy as np
import urllib.request
import kagglehub


# Function to load pre-trained TensorFlow object detection model
def load_model():
    # Example: Download and load a pre-trained model from TensorFlow Hub
    model_dir = "C:\\Users\\OmiTech27\\Desktop\\SEM4\\object_detection_project\\model"
    model = tf.saved_model.load(model_dir)
    return model

# Function to perform object detection on the image
def detect_objects_in_image(image_file):
    # Load pre-trained TensorFlow object detection model
    detection_model = load_model()

    # Read the image
    img = cv2.imread(image_file)

    # Convert image to RGB (TensorFlow object detection model expects RGB format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform object detection
    detections = detection_model(img_rgb)

    # Draw bounding boxes around detected objects
    processed_img = draw_boxes(img_rgb, detections)

    # Convert processed image back to BGR format for displaying with OpenCV
    processed_img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

    # Return processed image
    return processed_img_bgr

# Function to draw bounding boxes around detected objects
def draw_boxes(image, detections):
    # Example: Iterate through detections and draw bounding boxes
    for i in range(len(detections['detection_boxes'])):
        bbox = detections['detection_boxes'][i].numpy()
        bbox = [int(coord * image.shape[0]) for coord in bbox]
        cv2.rectangle(image, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (0, 255, 0), 2)
        class_name = detections['detection_classes'][i].numpy()
        # Example: Get class name from class index
        # class_name = class_names[class_index]
        cv2.putText(image, class_name, (bbox[1], bbox[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image
