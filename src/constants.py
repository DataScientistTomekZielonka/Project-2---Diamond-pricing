import json
import os

# Get the absolute path of the current file (constants.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to mappings.json
mappings_path = os.path.join(current_dir, '../data/mappings.json')

with open(mappings_path, 'r') as file :
    mappings = json.load(file)

CUT_MAPPING = mappings.get('cut_map', {})
COLOR_MAPPING = mappings.get('color_map', {})
CLARITY_MAPPING = mappings.get('clarity_map', {})