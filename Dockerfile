FROM python:3.9

RUN addgroup --system app && adduser --system --group app
USER app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY index.py .
COPY data/ ./data/
COPY images/ ./images/



ENTRYPOINT  ["python", "./index.py"]
