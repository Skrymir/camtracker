# camtracker: camera person tracker for an x86-64 server with a USB Coral.
# Build (CI publishes the same as ghcr.io/skrymir/camtracker):
#   docker build -t ghcr.io/skrymir/camtracker .
#
# Google's Coral apt repo is gone (the project was deleted), so this follows
# Frigate: Debian 12 (bookworm), libedgetpu from feranick's maintained builds,
# and the tflite_runtime wheel Frigate publishes. No pycoral. Both downloads
# are checked against their sha256.
FROM python:3.11-slim-bookworm

ARG LIBEDGETPU_URL=https://github.com/feranick/libedgetpu/releases/download/16.0TF2.17.1-1/libedgetpu1-std_16.0tf2.17.1-1.bookworm_amd64.deb
ARG LIBEDGETPU_SHA256=a3e1f9ae3e4725032a863ea38f52aa396ee18fdb65135fed56e8efae5c2e467d
ARG TFLITE_URL=https://github.com/frigate-nvr/TFlite-builds/releases/download/v2.17.1/tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl
ARG TFLITE_SHA256=a6b4e57f13c536a737aeb0ebd27618177ddbbabb1e4242c9c3403cd566fe71b6

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libusb-1.0-0 libglib2.0-0 \
 && curl -fsSL -o /tmp/libedgetpu.deb "$LIBEDGETPU_URL" \
 && echo "$LIBEDGETPU_SHA256  /tmp/libedgetpu.deb" | sha256sum -c - \
 && dpkg -i /tmp/libedgetpu.deb && rm /tmp/libedgetpu.deb \
 && rm -rf /var/lib/apt/lists/*

# Same paho as requirements.txt (tests/test_docker.py checks). numpy 1.x:
# tflite_runtime 2.17 is built against it.
RUN curl -fsSL -o /tmp/tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl "$TFLITE_URL" \
 && echo "$TFLITE_SHA256  /tmp/tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl" | sha256sum -c - \
 && pip install --no-cache-dir /tmp/tflite_runtime-2.17.1-cp311-cp311-linux_x86_64.whl \
        numpy==1.26.4 opencv-python-headless==4.10.0.84 paho-mqtt==2.1.0 \
 && rm /tmp/*.whl

# The detection model Frigate uses on a Coral (COCO; class 0 = person).
RUN mkdir -p /models && curl -fsSL -o /models/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite \
        https://github.com/google-coral/test_data/raw/release-frogfish/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite

WORKDIR /app
COPY camtracker.py status.py web.py ./
COPY camtracker.ini /app/camtracker.ini.example

# The live view ([web] port)
EXPOSE 8080

# Config lives on a volume: /config/camtracker.ini
ENV CAMTRACKER_CONFIG=/config/camtracker.ini PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/app/camtracker.py"]
