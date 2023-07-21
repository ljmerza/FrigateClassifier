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
  threshold: 0.7
dog_classification:
  threshold: 0.7
logger_level: INFO
```

Update your frigate url, mqtt server settings. If you are using mqtt authentication, update the username and password. Update the camera name(s) to match the camera name in your frigate config.

### Running

```bash
docker run -v /path/to/config:/config -e TZ=America/New_York -it --rm --name frigate_classifier lmerza/frigate_classifier:latest
```

or using docker-compose:

```yml
services:
  frigate_classifier:
    image: lmerza/frigate_classifier:latest
    container_name: frigate_classifier
    volumes:
      - /path/to/config:/config
    restart: unless-stopped
    environment:
      - TZ=America/New_York
```

### Debugging

set `logger_level` in your config to `DEBUG` to see more logging information:

```yml
...
logger_level: DEBUG
```

Logs will be in `/config/frigateclassifier.log`