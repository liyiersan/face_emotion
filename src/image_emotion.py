import sys
import cv2
from keras.models import load_model
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_image
from utils.preprocessor import preprocess_input
import time

start = time.process_time()

# parameters for loading data and images

image_path = sys.argv[1]
image_name = os.path.basename(image_path)
emotion_model_path = '../trained_models/fer2013_big_XCEPTION.54-0.66.hdf5'
emotion_labels = get_labels('fer2013')

# hyper-parameters for bounding boxes shape
emotion_offsets = (0, 0)


# loading models
emotion_classifier = load_model(emotion_model_path, compile=False)

# getting input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# loading images
rgb_image = load_image(image_path, grayscale=False)
gray_image = load_image(image_path, grayscale=True)
gray_image = np.squeeze(gray_image)
gray_image = gray_image.astype('uint8')

faces = detect_faces(gray_image)

for face_coordinates in faces:

    x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
    gray_face = gray_image[y1:y2, x1:x2]
    try:
        gray_face = cv2.resize(gray_face, (emotion_target_size))
    except:
        continue

    gray_face = preprocess_input(gray_face, True)
    gray_face = np.expand_dims(gray_face, 0)
    gray_face = np.expand_dims(gray_face, -1)
    emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
    emotion_text = emotion_labels[emotion_label_arg]
    color = (255, 0, 0)

    draw_bounding_box(face_coordinates, rgb_image, color)
    draw_text(face_coordinates, rgb_image, emotion_text, color, 0, -15, 2, 2)

bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
end = time.process_time()
print('total time: ', end-start)
cv2.imwrite('../results/predicted_' + image_name, bgr_image)
