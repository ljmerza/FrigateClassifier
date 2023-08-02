import signal
from datetime import datetime
from io import BytesIO

import sqlite3
import numpy as np
import time
import multiprocessing
import cv2
import logging
from collections import defaultdict
import re

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from tflite_support import metadata_schema_py_generated as _metadata_fb

import paho.mqtt.client as mqtt
import hashlib
import yaml
import sys
import json
import requests
from PIL import Image, ImageOps

config = None
firstmessage = True
_LOGGER = None

VERSION = '1.0.3'

CONFIG_PATH = './config/config.yml'
DB_PATH = './config/classifier.db'
BIRD_NAME_DB = './data/bird_names.db'
LOG_FILE = './config/frigateclassifier.log'

IMAGE_FILE_FULL = './images/fullsized.jpg'
IMAGE_FILE_CROPPED = './images/cropped.jpg'
IMAGE_FILE_PADDED = './images/padded.jpg'


LABELS = {
    'DOG': 'dog',
    'BIRD': 'bird',
}

class TimeoutError(Exception):
    pass

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def get_common_bird_name(scientific_name):
    conn = sqlite3.connect(BIRD_NAME_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT common_name FROM birdnames WHERE scientific_name = ?", (scientific_name,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        _LOGGER.warning(f"No common bird name for: {scientific_name}")
        return ""

def get_common_dog_name(display_name):
    # Use regex to find the part after the hyphen
    pattern = r'-(.+)'
    match = re.search(pattern, display_name)

    if match:
        # Extract the part after the hyphen and title case it
        title_cased_part = match.group(1).title()
        return title_cased_part

    return display_name

def image_manipulation(response_content, after_data):
    image = Image.open(BytesIO(response_content))
    image.save(IMAGE_FILE_FULL, format="JPEG")

    # crop the image and save
    bounding_box = after_data['box']
    region = after_data['region']
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

def classify(response_content, after_data):
    label = after_data['label']
    _LOGGER.debug(f"classifying image for a {label}")

    # format image for classification
    image, padded_image, cropped_image = image_manipulation(response_content, after_data)
    np_arr = np.array(padded_image)
    tensor_image = vision.TensorImage.create_from_array(np_arr)
    # tensor_image = vision.TensorImage.create_from_file(IMAGE_FILE_FULL)

    # get classifier file
    file_name = None
    if label == LABELS['BIRD']:
        file_name='data/bird_model.tflite'
    elif label == LABELS['DOG']:
        file_name='data/dog_model.tflite'
    else:
        _LOGGER.error(f"Unknown label: {label}")
        return None
    
    # generate classifier and classify with timeout
    base_options = core.BaseOptions(file_name=file_name, use_coral=False, num_threads=4)
    classification_options = processor.ClassificationOptions(max_results=1, score_threshold=0)
    options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
    classifier = vision.ImageClassifier.create_from_options(options)

    try:
        with timeout(seconds=30):
            result = classifier.classify(tensor_image)
            categories = result.classifications[0].categories
            _LOGGER.debug(f'classify categories: {categories}')
            return categories
    except TimeoutError:
        _LOGGER.error(f"TimeoutError classifying event: {frigate_event}")
        return None


def on_connect(client, userdata, flags, rc):
    _LOGGER.info("MQTT Connected")
    client.subscribe(config['frigate']['main_topic'] + "/events")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        _LOGGER.warning("Unexpected disconnection, trying to reconnect")
        while True:
            try:
                client.reconnect()
                break
            except Exception as e:
                _LOGGER.warning(f"Reconnection failed due to {e}, retrying in 60 seconds")
                time.sleep(60)
    else:
        _LOGGER.error("Expected disconnection")


def set_sublabel(frigate_url, frigate_event, sublabel):
    post_url = f"{frigate_url}/api/events/{frigate_event}/sub_label"
    _LOGGER.debug(f'sublabel: {sublabel}')
    _LOGGER.debug(f'sublabel url: {post_url}')

    # frigate limits sublabels to 20 characters currently
    if len(sublabel) > 20:
        sublabel = sublabel[:20]

    # Submit the POST request with the JSON payload
    payload = { "subLabel": sublabel }
    headers = { "Content-Type": "application/json" }
    response = requests.post(post_url, data=json.dumps(payload), headers=headers)

    # Check for a successful response
    if response.status_code == 200:
        _LOGGER.info(f"Sublabel set successfully to: {sublabel}")
    else:
        _LOGGER.error(f"Failed to set sublabel. Status code: {response.status_code}")


def on_message(client, userdata, message):
    global firstmessage
    if firstmessage:
        firstmessage = False
        _LOGGER.debug("skipping first message")
        return

    # get frigate event payload
    payload_dict = json.loads(message.payload)
    _LOGGER.debug(f'mqtt message: {payload_dict}')
    after_data = payload_dict.get('after', {})

    if not after_data['camera'] in config['frigate']['camera']:
        _LOGGER.debug(f"Skipping event: {after_data['id']} because it is from the wrong camera: {after_data['camera']}")
        return

    is_bird = after_data['label'] == LABELS['BIRD']
    is_dog = after_data['label'] == LABELS['DOG']

    # get classification config
    classification_config = None
    if is_bird:
        classification_config = config.get('bird_classification')
    elif is_dog:
        classification_config = config.get('dog_classification')
    else:
        _LOGGER.debug(f"Skipping event: {after_data['id']} because it is not a classified object: {after_data['label']}")
        return
    if not classification_config:
        _LOGGER.error(f"Could not find classification config for {after_data['label']}")
        return
    
    # get frigate event
    frigate_event = after_data['id']
    frigate_url = config['frigate']['frigate_url']

    snapshot_url = f"{frigate_url}/api/events/{frigate_event}/snapshot.jpg"
    _LOGGER.debug(f"Getting image for event: {frigate_event}" )
    _LOGGER.debug(f"event URL: {snapshot_url}")

    response = requests.get(snapshot_url, params={ "crop": 1, "quality": 95 })

    # Check if the request was successful (HTTP status code 200)
    if response.status_code != 200:
        _LOGGER.error(f"Error getting snapshot: {response.status_code}")
        return
    
    # classify  
    categories = classify(response.content, after_data)
    if categories is None:
        return
    
    # gather classification data
    category = categories[0]
    index = category.index
    score = category.score
    display_name = category.display_name or category.category_name 
    category_name = category.category_name

    start_time = datetime.fromtimestamp(after_data['start_time'])
    formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    result_text = formatted_start_time + "\n"
    result_text = result_text + str(category)
    _LOGGER.debug(f"result_text: {result_text}")

    # check threshold or background event
    if index == 964 or score < classification_config['threshold']:  # 964 is "background"
        _LOGGER.debug(f"Skipping event: {frigate_event} because it is below the threshold: {score}")
        return

    # get db connection
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if a record with the given frigate_event exists
    cursor.execute("SELECT * FROM detections WHERE frigate_event = ?", (frigate_event,))
    result = cursor.fetchone()

    # get sublabel that will be used
    sublabel = None
    if is_dog:
        sublabel = get_common_dog_name(display_name)
    elif is_bird:
        sublabel = get_common_bird_name(display_name)
    else:
        _LOGGER.error(f"Unknown label: {label}")
        return

    if result is None:
        # Insert a new record if it doesn't exist
        _LOGGER.info("No record yet for this event. Storing.")

        cursor.execute("""  
            INSERT INTO detections (detection_time, detection_index, score,  
            display_name, category_name, frigate_event, camera_name) VALUES (?, ?, ?, ?, ?, ?, ?)  
            """, (formatted_start_time, index, score, display_name, category_name, frigate_event, after_data['camera']))

        # set the sublabel
        set_sublabel(frigate_url, frigate_event, sublabel)
    else:
        _LOGGER.info("There is already a record for this event. Checking score")

        # Update the existing record if the new score is higher
        existing_score = result[3]

        if score > existing_score:
            _LOGGER.info("New score is higher. Updating record with higher score.")
            cursor.execute("""  
                UPDATE detections  
                SET detection_time = ?, detection_index = ?, score = ?, display_name = ?, category_name = ?  
                WHERE frigate_event = ?  
                """, (formatted_start_time, index, score, display_name, category_name, frigate_event))
            # set the sublabel
            set_sublabel(frigate_url, frigate_event, sublabel)
        else:
            _LOGGER.info("New score is lower.")

    # Commit the changes
    conn.commit()
    conn.close()


def setup_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""    
        CREATE TABLE IF NOT EXISTS detections (    
            id INTEGER PRIMARY KEY AUTOINCREMENT,  
            detection_time TIMESTAMP NOT NULL,  
            detection_index INTEGER NOT NULL,  
            score REAL NOT NULL,  
            display_name TEXT NOT NULL,  
            category_name TEXT NOT NULL,  
            frigate_event TEXT NOT NULL UNIQUE,
            camera_name TEXT NOT NULL 
        )    
    """)
    conn.commit()
    conn.close()

def load_config():
    global config
    with open(CONFIG_PATH, 'r') as config_file:
        config = yaml.safe_load(config_file)

def run_mqtt_client():
    _LOGGER.info(f"Starting MQTT client. Connecting to: {config['frigate']['mqtt_server']}")
    now = datetime.now()
    current_time = now.strftime("%Y%m%d%H%M%S")

    # setup mqtt client
    client = mqtt.Client("FrigateClassifier" + current_time)
    client.on_message = on_message
    client.on_disconnect = on_disconnect
    client.on_connect = on_connect

    # check if we are using authentication and set username/password if so
    if config['frigate']['mqtt_auth']:
        username = config['frigate']['mqtt_username']
        password = config['frigate']['mqtt_password']
        client.username_pw_set(username, password)

    client.connect(config['frigate']['mqtt_server'])
    client.loop_forever()

def load_logger():
    global _LOGGER
    _LOGGER = logging.getLogger(__name__)
    _LOGGER.setLevel(config['logger_level'])

    # Create a formatter to customize the log message format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a console handler and set the level to display all messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    _LOGGER.addHandler(console_handler)
    _LOGGER.addHandler(file_handler)
    

def main():
    load_config()
    setup_db()
    load_logger()

    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    _LOGGER.info(f"Time: {current_time}")
    _LOGGER.info(f"Python Version: {sys.version}")
    _LOGGER.info(f"Frigate Classifier Version: {VERSION}")
    _LOGGER.debug(f"config: {config}")

    # start mqtt client
    mqtt_process = multiprocessing.Process(target=run_mqtt_client)
    mqtt_process.start()
    mqtt_process.join()


if __name__ == '__main__':
    main()
