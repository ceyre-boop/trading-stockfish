"""
Replay event loader for validation lab.
Loads events from JSON files for replay.
"""

import json


def load_replay_events(path):
    with open(path, "r") as f:
        return json.load(f)
