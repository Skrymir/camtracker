"""The image installs the paho in requirements.txt and copies every module
camtracker.py imports."""
import re
import xml.etree.ElementTree as ET
from pathlib import Path


def test_dockerfile_paho_matches_requirements():
    pin = re.search(r"^paho-mqtt==(\S+)", Path("requirements.txt").read_text(), re.M).group(1)
    assert f"paho-mqtt=={pin}" in Path("Dockerfile").read_text()


def test_dockerfile_copies_what_camtracker_imports():
    df, ignore = Path("Dockerfile").read_text(), Path(".dockerignore").read_text()
    for f in ("camtracker.py", "status.py", "web.py", "web.html"):
        assert f in df and f"!{f}" in ignore


def test_unraid_template_runs_like_compose():
    t = ET.parse("unraid/camtracker.xml").getroot()
    compose = Path("docker-compose.yml").read_text()
    image = t.findtext("Repository").split(":")[0]
    assert f"image: {image}" in compose and image in Path("README.md").read_text()
    assert "c 189:* rmw" in t.findtext("ExtraParams") and '"c 189:* rmw"' in compose
    paths = {c.get("Target"): c.text for c in t.iter("Config") if c.get("Type") == "Path"}
    assert paths == {"/config": "/mnt/user/appdata/camtracker", "/dev/bus/usb": "/dev/bus/usb"}
    ports = {c.get("Target") for c in t.iter("Config") if c.get("Type") == "Port"}
    assert ports == {"8080"} and "[PORT:8080]" in t.findtext("WebUI")
    assert '"8080:8080"' in compose and "EXPOSE 8080" in Path("Dockerfile").read_text()
    env = {c.get("Target") for c in t.iter("Config") if c.get("Type") == "Variable"}
    assert env == {"CAMERA_URL", "MQTT_USERNAME", "MQTT_PASSWORD"}
