import sqlite3
import numpy as np
from datetime import datetime
import time
import multiprocessing
import cv2
import logging
from collections import defaultdict

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
from io import BytesIO

bird_classifier = None
dog_classifier = None
config = None
firstmessage = True
_LOGGER = None

VERSION = '1.0.1-beta'

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


def get_common_bird_name(scientific_name):
    conn = sqlite3.connect(BIRD_NAME_DB)
    cursor = conn.cursor()

    cursor.execute("SELECT common_name FROM birdnames WHERE scientific_name = ?", (scientific_name,))
    result = cursor.fetchone()
    conn.close()

    if result:
        return result[0]
    else:
        _LOGGER.warn(f"No common bird name for: {scientific_name}")
        return ""


def image_manipulation(response_content, after_dataafter_data):
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
    categories = None
    label = after_data['label']
    _LOGGER.debug(f"classifying image for a {label}")

    image, padded_image, cropped_image = image_manipulation(response_content, after_data)

    np_arr = np.array(padded_image)
    tensor_image = vision.TensorImage.create_from_array(np_arr)
    # tensor_image = vision.TensorImage.create_from_file(IMAGE_FILE_FULL)

    if label == LABELS['BIRD']:
        categories = bird_classifier.classify(tensor_image)
    elif label == LABELS['DOG']:
        categories = dog_classifier.classify(tensor_image)
    else:
        _LOGGER.error(f"Unknown label: {label}")
        return None
    
    _LOGGER.debug(f'classify categories: {categories}')
    return categories.classifications[0].categories


def on_connect(client, userdata, flags, rc):
    _LOGGER.info("MQTT Connected")
    client.subscribe(config['frigate']['main_topic'] + "/events")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        _LOGGER.warn("Unexpected disconnection, trying to reconnect")
        while True:
            try:
                client.reconnect()
                break
            except Exception as e:
                _LOGGER.warn(f"Reconnection failed due to {e}, retrying in 60 seconds")
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

    payload = { "subLabel": sublabel }

    # Set the headers for the request
    headers = { "Content-Type": "application/json" }

    # Submit the POST request with the JSON payload
    response = requests.post(post_url, data=json.dumps(payload), headers=headers)

    # Check for a successful response
    if response.status_code == 200:
        _LOGGER.info(f"Sublabel set successfully to: {sublabel}")
    else:
        _LOGGER.error(f"Failed to set sublabel. Status code: {response.status_code}")


def on_message(client, userdata, message):
    conn = sqlite3.connect(DB_PATH)

    global firstmessage
    if not firstmessage:

        # Convert the MQTT payload to a Python dictionary
        payload_dict = json.loads(message.payload)
        _LOGGER.debug(f'mqtt message: {payload_dict}')

        # Extract the 'after' element data and store it in a dictionary
        after_data = payload_dict.get('after', {})

        is_bird = after_data['label'] == LABELS['BIRD']
        is_dog = after_data['label'] == LABELS['DOG']

        is_classified_object = is_bird or is_dog
        classification_config = config['bird_classification'] if is_bird else config['dog_classification']
        
        if (after_data['camera'] in config['frigate']['camera'] and is_classified_object):
            frigate_event = after_data['id']
            frigate_url = config['frigate']['frigate_url']

            snapshot_url = f"{frigate_url}/api/events/{frigate_event}/snapshot.jpg"
            _LOGGER.debug(f"Getting image for event: {frigate_event}" )
            _LOGGER.debug(f"event URL: {snapshot_url}")

            response = requests.get(snapshot_url, params={ "crop": 1, "quality": 95 })

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                categories = classify(response.content, after_data)
                if categories is None:
                    return
                    
                category = categories[0]
                index = category.index
                score = category.score
                display_name = category.display_name
                category_name = category.category_name

                start_time = datetime.fromtimestamp(after_data['start_time'])
                formatted_start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
                result_text = formatted_start_time + "\n"
                result_text = result_text + str(category)
                _LOGGER.debug(f"result_text: {result_text}")

                if index != 964 and score > classification_config['threshold']:  # 964 is "background"
                    cursor = conn.cursor()

                    # Check if a record with the given frigate_event exists
                    cursor.execute("SELECT * FROM detections WHERE frigate_event = ?", (frigate_event,))
                    result = cursor.fetchone()

                    name = get_common_bird_name(display_name) or display_name if is_bird else display_name

                    if result is None:
                        # Insert a new record if it doesn't exist
                        _LOGGER.info("No record yet for this event. Storing.")

                        cursor.execute("""  
                            INSERT INTO detections (detection_time, detection_index, score,  
                            display_name, category_name, frigate_event, camera_name) VALUES (?, ?, ?, ?, ?, ?, ?)  
                            """, (formatted_start_time, index, score, display_name, category_name, frigate_event, after_data['camera']))

                        # set the sublabel
                        set_sublabel(frigate_url, frigate_event, name)
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
                            set_sublabel(frigate_url, frigate_event, name)
                        else:
                            _LOGGER.info("New score is lower.")

                    # Commit the changes
                    conn.commit()

            else:
                _LOGGER.error(f"Error: Could not retrieve the image: {response.text}")
        else:
            if after_data['camera'] not in config['frigate']['camera']:
                _LOGGER.debug(f"Skipping event: {after_data['id']} because it is from the wrong camera: {after_data['camera']}")
            else:
                _LOGGER.debug(f"Skipping event: {after_data['id']} because it is not a classified object: {after_data['label']}")
    else:
        firstmessage = False
        _LOGGER.debug("skipping first message")

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

    classification_options = processor.ClassificationOptions(max_results=1, score_threshold=0)

    # Initialize the image classification model for birds and create classifier
    base_options = core.BaseOptions(file_name='data/bird_model.tflite', use_coral=False, num_threads=4)
    options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
    global bird_classifier
    bird_classifier = vision.ImageClassifier.create_from_options(options)

    # Initialize the image classification model for dog and create classifier
    base_options = core.BaseOptions(file_name='data/dog_model.tflite', use_coral=False, num_threads=4)
    options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
    global dog_classifier
    dog_classifier = vision.ImageClassifier.create_from_options(options)

    # start mqtt client
    mqtt_process = multiprocessing.Process(target=run_mqtt_client)
    mqtt_process.start()
    mqtt_process.join()


if __name__ == '__main__':
    main()
