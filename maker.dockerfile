# FROM python:latest
# RUN pip install tensorflow numpy
# RUN pip install -q tflite-model-maker

FROM waikatodatamining/tflite_model_maker:2.4.3

WORKDIR /usr/app/src

COPY Images .
COPY model_maker.py .

ENTRYPOINT ["python", "model_maker.py"]