"""Reading-room theme: paper-grey/navy/highlighter-yellow, Newsreader + Public Sans."""
import streamlit as st

BG = "#EBEDEF"
BG_SECONDARY = "#DDE0E3"
INK = "#1B2430"
ACCENT = "#F4E04D"
LINK = "#4C6FA5"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,wght@0,400;0,500;0,600;1,400&family=Public+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Public Sans', sans-serif;
    color: {INK};
}}

h1, h2, h3 {{
    font-family: 'Newsreader', serif;
}}

.stApp {{ background-color: {BG}; }}

a, a:visited {{ color: {LINK}; }}

.paper-card {{
    background: white;
    border: 1px solid {BG_SECONDARY};
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}}

.source-badge {{
    display: inline-block;
    background: {BG_SECONDARY};
    color: {LINK};
    border-radius: 999px;
    padding: 0.1rem 0.6rem;
    font-size: 0.75rem;
    margin-right: 0.3rem;
}}

.oa-marker {{
    display: inline-block;
    color: #2f6f4f;
    font-size: 0.75rem;
    font-weight: 600;
}}

.ai-summary {{
    border-left: 4px solid {ACCENT};
    background: #FFFDF0;
    padding: 0.5rem 0.9rem;
    margin-top: 0.5rem;
}}

.citation-marker {{
    color: {INK};
    background: {ACCENT};
    border-radius: 3px;
    padding: 0 0.2rem;
    font-weight: 600;
}}
</style>
"""


def inject():
    st.markdown(CSS, unsafe_allow_html=True)
