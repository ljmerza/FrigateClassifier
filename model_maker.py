import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader


data = DataLoader.from_folder('./Images')

train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

model = image_classifier.create(train_data, validation_data=validation_data)
model.summary()

loss, accuracy = model.evaluate(test_data)

model.export(export_dir='./Images')

"""
docker build -f maker.dockerfile . -t lmerza/frigateclassifier_mm:latest
docker run -v ./Images:/usr/app/src/Images lmerza/frigateclassifier_mm:latest

docker build -t lmerza/frigateclassifier:latest . 
docker login
docker tag lmerza/frigateclassifier:latest lmerza/frigateclassifier:latest
docker push lmerza/frigateclassifier:latest
docker system prune -a
"""