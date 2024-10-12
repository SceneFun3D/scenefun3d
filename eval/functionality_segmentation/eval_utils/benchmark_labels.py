import numpy as np

CLASS_LABELS = [
    "rotate",
    "key_press",
    "tip_push",
    "hook_pull",
    "pinch_pull",
    "hook_turn",
    "foot_push",
    "plug_in",
    "unplug",
]

VALID_CLASS_IDS = np.array(
    # [2, 3, 4, 5, 6, 7, 8, 9, 10]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
)

EXCLUDE_ID = 255