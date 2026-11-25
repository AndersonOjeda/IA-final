"""Gestor de almacenamiento y particionado de anotaciones de residuos.

Soporta:
- Asignación de una de las 6 clases del proyecto:
  plastico, papel, carton, vidrio, metal, organicos.
- Recuadros (x1, y1, x2, y2) por objeto en píxeles normalizados o absolutos.
- Fotogramas vacíos revisados como ejemplos negativos sin residuos.
- Metadatos de procedencia: video_id, session_id, frame_id, split (train/val/test).
- Partición estricta por sesión/video para prevenir fugas de datos (data leakage).
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

CLASSES = ["plastico", "papel", "carton", "vidrio", "metal", "organicos"]

CLASS_DISPLAY = {
    "plastico": "Plástico",
    "papel": "Papel",
    "carton": "Cartón",
    "vidrio": "Vidrio",
    "metal": "Metal",
    "organicos": "Restos Orgánicos"
}

CLASS_TO_ID = {cls: idx for idx, cls in enumerate(CLASSES)}
ID_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}


class AnnotationRecord:
    def __init__(
        self,
        image_path: str,
        video_id: str,
        session_id: str,
        frame_index: int,
        boxes: Optional[List[Dict[str, Any]]] = None,
        is_empty_reviewed: bool = False,
        split: str = "train",
        width: int = 640,
        height: int = 480
    ):
        self.image_path = str(image_path)
        self.video_id = str(video_id)
        self.session_id = str(session_id)
        self.frame_index = int(frame_index)
        self.boxes = boxes or []
        self.is_empty_reviewed = bool(is_empty_reviewed)
        self.split = str(split)
        self.width = int(width)
        self.height = int(height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_path": self.image_path,
            "video_id": self.video_id,
            "session_id": self.session_id,
            "frame_index": self.frame_index,
            "boxes": self.boxes,
            "is_empty_reviewed": self.is_empty_reviewed,
            "split": self.split,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationRecord":
        return cls(
            image_path=data["image_path"],
            video_id=data["video_id"],
            session_id=data.get("session_id", data["video_id"]),
            frame_index=data.get("frame_index", 0),
            boxes=data.get("boxes", []),
            is_empty_reviewed=data.get("is_empty_reviewed", False),
            split=data.get("split", "train"),
            width=data.get("width", 640),
            height=data.get("height", 480)
        )


class AnnotationDataset:
    def __init__(self, file_path: str = "data/annotations/annotations.json"):
        self.file_path = Path(file_path)
        self.records: Dict[str, AnnotationRecord] = {}
        self.load()

    def load(self):
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data.get("records", []):
                    rec = AnnotationRecord.from_dict(item)
                    self.records[rec.image_path] = rec

    def save(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "classes": CLASSES,
            "records": [rec.to_dict() for rec in self.records.values()]
        }
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_or_update(
        self,
        image_path: str,
        video_id: str,
        session_id: str,
        frame_index: int,
        boxes: List[Dict[str, Any]],
        is_empty_reviewed: bool = False,
        split: Optional[str] = None,
        width: int = 640,
        height: int = 480
    ) -> AnnotationRecord:
        # Validar cajas
        clean_boxes = []
        for b in boxes:
            cls = b.get("label", "").lower()
            if cls not in CLASSES:
                raise ValueError(f"Clase desconocida '{cls}'. Permitidas: {CLASSES}")
            x1 = float(b["x1"])
            y1 = float(b["y1"])
            x2 = float(b["x2"])
            y2 = float(b["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            clean_boxes.append({
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
                "label": cls
            })

        if len(clean_boxes) > 0:
            is_empty_reviewed = False

        existing = self.records.get(image_path)
        current_split = split or (existing.split if existing else "train")

        rec = AnnotationRecord(
            image_path=image_path,
            video_id=video_id,
            session_id=session_id,
            frame_index=frame_index,
            boxes=clean_boxes,
            is_empty_reviewed=is_empty_reviewed,
            split=current_split,
            width=width,
            height=height
        )
        self.records[image_path] = rec
        self.save()
        return rec

    def get(self, image_path: str) -> Optional[AnnotationRecord]:
        return self.records.get(image_path)

    def delete(self, image_path: str) -> bool:
        if image_path in self.records:
            del self.records[image_path]
            self.save()
            return True
        return False

    def split_by_session(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
    ) -> Dict[str, List[str]]:
        """Particiona asignando sesiones enteras a train, val o test.
        Garantiza que ningún fotograma de la misma sesión/grabación se mezcle entre conjuntos.
        """
        import random
        sessions = sorted(list({rec.session_id for rec in self.records.values()}))
        rng = random.Random(seed)
        rng.shuffle(sessions)

        n = len(sessions)
        if n == 0:
            return {"train": [], "val": [], "test": []}
        if n == 1:
            # Una sola sesión: advertencia
            split_map = {sessions[0]: "train"}
        elif n == 2:
            split_map = {sessions[0]: "train", sessions[1]: "val"}
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            if n_train + n_val >= n:
                n_train = max(1, n - 2)
                n_val = 1
            train_sessions = set(sessions[:n_train])
            val_sessions = set(sessions[n_train:n_train + n_val])
            test_sessions = set(sessions[n_train + n_val:])
            split_map = {}
            for s in train_sessions:
                split_map[s] = "train"
            for s in val_sessions:
                split_map[s] = "val"
            for s in test_sessions:
                split_map[s] = "test"

        for rec in self.records.values():
            rec.split = split_map.get(rec.session_id, "train")

        self.save()
        return {
            "train": [s for s, sp in split_map.items() if sp == "train"],
            "val": [s for s, sp in split_map.items() if sp == "val"],
            "test": [s for s, sp in split_map.items() if sp == "test"]
        }

    def summary(self) -> Dict[str, Any]:
        """Calcula el resumen de videos, imágenes y objetos por clase y partición."""
        from collections import Counter

        videos = set()
        sessions = set()
        split_images = Counter()
        split_reviewed_empty = Counter()
        split_pending = 0
        objects_by_split_and_class = {
            "train": Counter(),
            "val": Counter(),
            "test": Counter()
        }

        for rec in self.records.values():
            videos.add(rec.video_id)
            sessions.add(rec.session_id)
            sp = rec.split if rec.split in ("train", "val", "test") else "train"
            split_images[sp] += 1
            if rec.is_empty_reviewed and len(rec.boxes) == 0:
                split_reviewed_empty[sp] += 1
            for b in rec.boxes:
                objects_by_split_and_class[sp][b["label"]] += 1

        return {
            "total_videos": len(videos),
            "total_sessions": len(sessions),
            "total_images": len(self.records),
            "split_images": dict(split_images),
            "split_reviewed_empty": dict(split_reviewed_empty),
            "objects_per_class": objects_by_split_and_class
        }
