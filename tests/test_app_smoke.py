from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parent.parent / "app.py")


def test_app_boots_without_error():
    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
