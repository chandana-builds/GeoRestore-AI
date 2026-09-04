import os
import sys

# Ensure root and app directories are on sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(root_dir, "app")
for path in [root_dir, app_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Run the Streamlit application
from app.app import *
