import tensorflow as tf
import cv2
import numpy as np
import os
from PIL import Image
from pillow_heif import register_heif_opener

register_heif_opener()


# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="hidey_camguard.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
test_folder = os.path.join(os.getcwd(), 'datasets/no_camera_present')

for img_name in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_name)

    if img_path.lower().endswith('.heic'):
        image = Image.open(img_path)
        # Convert to JPEG
        img_path = img_path.split(".")[0] + '.jpg'
        image.convert('RGB').save(img_path)

    img = cv2.imread(img_path)

    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"For image: {img_name}, model output is: {output_data}")

    # Interpret the output (you can adjust this based on how you've trained your model)
    if output_data[0][0] > output_data[0][1]:
        print("Model predicts: No camera present")
    else:
        print("Model predicts: Camera present")
