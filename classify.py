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
IMAGE_FILE_FULL = './fullsized_test.jpg'
IMAGE_FILE_CROPPED = './cropped_test.jpg'

config = None
with open(CONFIG_PATH, 'r') as config_file:
  config = yaml.safe_load(config_file)


frigate_url = config['frigate']['frigate_url']
frigate_event = '1689904097.55648-4d2bwl'
crop = (184, 212, 233, 275)
region = (51, 40, 371, 360)

def add_lists_elementwise(list1, list2):
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")
    result = (list1[i] + list2[i] for i in range(len(list1)))
    return result

crop_transform = region # add_lists_elementwise(crop, region)

if not frigate_event:
  raise OSError('No event specified')

snapshot_url = f"{frigate_url}/api/events/{frigate_event}/snapshot.jpg"

params = {
    # "crop": 1,
    "quality": 95
}
response = requests.get(snapshot_url, params=params)

if response.status_code != 200:
  raise OSError(f"Error getting snapshot: {response.status_code}")

image = Image.open(BytesIO(response.content))
image.save(IMAGE_FILE_FULL, format="JPEG")

# Resize the image while maintaining its aspect ratio
# max_size = (224, 224)
# image.thumbnail(max_size)

# Pad the image to fill the remaining space
padded_image = image.crop(crop_transform)
padded_image.save(IMAGE_FILE_CROPPED, format="JPEG")

classification_options = processor.ClassificationOptions(max_results=1, score_threshold=0)

# Initialization
base_options = core.BaseOptions(file_name='data/bird_model.tflite', use_coral=False, num_threads=4)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
bird_classifier = vision.ImageClassifier.create_from_options(options)

base_options = core.BaseOptions(file_name='data/dog_model.tflite', use_coral=False, num_threads=4)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
dog_classifier = vision.ImageClassifier.create_from_options(options)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

def get_specs(result):
  categories = result.classifications[0].categories
  category = categories[0]
  index = category.index
  score = category.score
  display_name = category.display_name
  category_name = category.category_name

  print(f"result_text: {category_name}, {display_name}, {score}, {index}")

image = vision.TensorImage.create_from_file(IMAGE_FILE_FULL)
classification_result_bird = bird_classifier.classify(image)
classification_result_dog = dog_classifier.classify(image)
get_specs(classification_result_bird)
get_specs(classification_result_dog)

image = vision.TensorImage.create_from_file(IMAGE_FILE_CROPPED)
classification_result_bird = bird_classifier.classify(image)
classification_result_dog = dog_classifier.classify(image)
get_specs(classification_result_bird)
get_specs(classification_result_dog)


interpreter = tflite.Interpreter(model_path='data/bird_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread(IMAGE_FILE_FULL)
img = cv2.resize(img,(224,224))
input_shape = input_details[0]['shape']
input_tensor= np.array(np.expand_dims(img,0))

input_index = interpreter.get_input_details()[0]["index"]
interpreter.set_tensor(input_index, input_tensor)
interpreter.invoke()
output_details = interpreter.get_output_details()

output_data = interpreter.get_tensor(output_details[0]['index'])
pred = np.squeeze(output_data)
highest_pred_loc = np.argmax(pred)
print(highest_pred_loc)
print(output_details)