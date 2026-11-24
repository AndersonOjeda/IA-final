"""Confirmación temporal: el diagnóstico conserva siempre la predicción CNN original."""
from collections import Counter, deque
from dataclasses import replace

import cv2
import numpy as np

from app.vision import crop_region


class PredictionFilter:
    def __init__(self):
        self.readings = deque(maxlen=3)
        self.previous_crop = None

    def reset(self):
        self.readings.clear()
        self.previous_crop = None

    def update(self, frame, reading):
        crop = cv2.resize(crop_region(frame, reading.region), (24, 24)).astype(np.float32)
        if self.previous_crop is not None and np.abs(crop-self.previous_crop).mean() > 25:
            self.reset()
        self.previous_crop = crop
        self.readings.append(reading)
        label, votes = Counter(item.material for item in self.readings).most_common(1)[0]
        if votes < 2:
            return None
        probabilities = {key: sum(item.probabilities[key] for item in self.readings)/len(self.readings)
                         for key in reading.probabilities}
        return replace(reading, material=label, confidence=probabilities[label], probabilities=probabilities)
