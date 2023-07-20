# Frigate Classifier

This is a docker image that runs a python script that can be used to identify species/breeds of animals (birds and dogs) detected by [blakeblackshear/frigate](https://github.com/blakeblackshear/frigate)


### Setup

Create a `config.yml` file in your docker volume with the following contents:

```yml
frigate:
  frigate_url: http://127.0.0.1:5000
  mqtt_server: 127.0.0.1
  mqtt_auth: false
  mqtt_username: username
  mqtt_password: password
  main_topic: frigate
  camera:
    - birdcam
bird_classification:
  model: data/bird_model.tflite
  threshold: 0.7
dog_classification:
  model: data/dog_model.tflite
  threshold: 0.7
logger_level: INFO
```

### Running

```bash
docker run -v /path/to/config.yml:config.yml -v /path/to/data:/data -e TZ=America/New_York -it --rm --name frigate_classifier lmerza/frigate_classifier:latest
```

or using docker-compose:

```yml
services:
  frigate_classifier:
    image: lmerza/frigate_classifier:latest
    container_name: frigate_classifier
    volumes:
      - /path/to/config.yml:/config.yml
      - /path/to/data:/data
    restart: unless-stopped
    environment:
      - TZ=America/New_York
```