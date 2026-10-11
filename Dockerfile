# camtracker: camera person tracker for an x86-64 server with a USB Coral.
# Build:
#   docker build -t camtracker .
#
# Debian 11 (bullseye) on purpose: Google's Coral packages (libedgetpu,
# pycoral) are no longer updated and only install cleanly on bullseye's
# Python 3.9 with its numpy 1.x. Everything here comes from apt except paho.
FROM debian:bullseye-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
 && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/coral-edgetpu.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        > /etc/apt/sources.list.d/coral-edgetpu.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        libedgetpu1-std python3-pycoral python3-opencv python3-numpy python3-pip \
 && rm -rf /var/lib/apt/lists/*

# Same paho as requirements.txt (tests/test_docker.py checks).
RUN pip3 install --no-cache-dir paho-mqtt==2.1.0

# The detection model Frigate uses on a Coral (COCO; class 0 = person).
RUN mkdir -p /models && curl -fsSL -o /models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite \
        https://github.com/google-coral/test_data/raw/release-frogfish/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite

WORKDIR /app
COPY camtracker.py status.py ./
COPY camtracker.ini /app/camtracker.ini.example

# Config lives on a volume: /config/camtracker.ini
ENV CAMTRACKER_CONFIG=/config/camtracker.ini PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/app/camtracker.py"]
