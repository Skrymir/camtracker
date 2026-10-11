"""The image installs the paho in requirements.txt and copies every module
camtracker.py imports."""
import re
from pathlib import Path


def test_dockerfile_paho_matches_requirements():
    pin = re.search(r"^paho-mqtt==(\S+)", Path("requirements.txt").read_text(), re.M).group(1)
    assert f"paho-mqtt=={pin}" in Path("Dockerfile").read_text()


def test_dockerfile_copies_what_camtracker_imports():
    df, ignore = Path("Dockerfile").read_text(), Path(".dockerignore").read_text()
    for f in ("camtracker.py", "status.py"):
        assert f in df and f"!{f}" in ignore
