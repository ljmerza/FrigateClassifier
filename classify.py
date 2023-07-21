from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor

import tensorflow.lite as tflite

import requests
import yaml
from io import BytesIO
from PIL import Image, ImageOps
import cv2
import numpy as np


CONFIG_PATH = './config/config.yml'
IMAGE_FILE_FULL = './images/fullsized_test.jpg'
IMAGE_FILE_CROPPED = './images/cropped_test.jpg'
IMAGE_FILE_PADDED = './images/padded_test.jpg'

config = None
with open(CONFIG_PATH, 'r') as config_file:
  config = yaml.safe_load(config_file)


frigate_url = config['frigate']['frigate_url']
frigate_event = '1689958086.071282-gqgpdf'
bounding_box = [1341, 895, 1495, 1061]
region = [1155, 496, 1739, 1080]

if not frigate_event:
  raise OSError('No event specified')

snapshot_url = f"{frigate_url}/api/events/{frigate_event}/snapshot.jpg"

params = {
    "crop": 1,
    "quality": 95
}
response = requests.get(snapshot_url, params=params)

if response.status_code != 200:
  raise OSError(f"Error getting snapshot: {response.status_code}")

classification_options = processor.ClassificationOptions(max_results=1, score_threshold=0)

# Initialization
base_options = core.BaseOptions(file_name='data/bird_model.tflite', use_coral=False, num_threads=4)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
bird_classifier = vision.ImageClassifier.create_from_options(options)

base_options = core.BaseOptions(file_name='data/dog_model.tflite', use_coral=False, num_threads=4)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
dog_classifier = vision.ImageClassifier.create_from_options(options)


def image_manipulation(response_content):
    image = Image.open(BytesIO(response_content))
    image.save(IMAGE_FILE_FULL, format="JPEG")

    # crop the image and save
    cropped_image = image.crop(bounding_box)
    cropped_image.save(IMAGE_FILE_CROPPED, format="JPEG")

    # Resize the image while maintaining its aspect ratio
    max_size = (224, 224)
    image.thumbnail(max_size)

    # Pad the image to fill the remaining space
    x = (max_size[0] - image.size[0]) // 2
    y = (max_size[1] - image.size[1]) // 2
    padded_image = ImageOps.expand(image, border=(x, y), fill='black')
    padded_image.save(IMAGE_FILE_PADDED, format="JPEG")

    return image, padded_image, cropped_image


def get_specs(result):
  categories = result.classifications[0].categories
  category = categories[0]
  index = category.index
  score = category.score
  display_name = category.display_name
  category_name = category.category_name

  print(f"result_text: {category_name}, {display_name}, {score}, {index}")


image, padded_image, cropped_image = image_manipulation(response.content)

np_arr = np.array(padded_image)
tensor_image = vision.TensorImage.create_from_array(np_arr)

classification_result_bird = bird_classifier.classify(tensor_image)
classification_result_dog = dog_classifier.classify(tensor_image)
get_specs(classification_result_bird)
get_specs(classification_result_dog)

tensor_image = vision.TensorImage.create_from_file(IMAGE_FILE_PADDED)
classification_result_bird = bird_classifier.classify(tensor_image)
classification_result_dog = dog_classifier.classify(tensor_image)
get_specs(classification_result_bird)
get_specs(classification_result_dog)