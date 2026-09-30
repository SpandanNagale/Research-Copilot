"""Secret lookup: st.secrets first, then env/.env. Only module in core/ allowed to import streamlit."""
import os

from dotenv import load_dotenv

load_dotenv()

try:
    import streamlit as st
    _HAS_STREAMLIT_SECRETS = True
except ImportError:
    st = None
    _HAS_STREAMLIT_SECRETS = False


def get_secret(name: str, default: str | None = None) -> str | None:
    if _HAS_STREAMLIT_SECRETS:
        try:
            if name in st.secrets:
                return st.secrets[name]
        except Exception:
            pass
    return os.environ.get(name, default)
