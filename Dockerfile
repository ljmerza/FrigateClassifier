FROM python:3.9
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY index.py .
COPY data/ ./data/
COPY images/ ./images/

CMD python ./index.py
