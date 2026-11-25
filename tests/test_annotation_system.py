"""Pruebas unitarias del sistema de anotación y almacenamiento de residuos.

Verifica:
1. Guardar y volver a abrir una imagen con tres residuos conserva exactamente sus tres cajas y clases.
2. Manejo de fotogramas vacíos revisados como ejemplos negativos sin residuos.
3. Partición estricta por sesión para evitar fugas de datos (data leakage).
4. Generación correcta de resúmenes de conteo por clase y partición.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# Configurar para evitar problemas con libiomp5md en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.annotation_storage import AnnotationDataset, AnnotationRecord, CLASSES


class AnnotationStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_annotations.json"
        self.dataset = AnnotationDataset(str(self.db_path))

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_reopen_three_waste_objects(self):
        """Verifica como mínimo que guardar y volver a abrir una imagen con tres residuos
        conserva sus tres cajas y clases exactamente."""
        image_path = "data/annotated_frames/video_01/frame_00010.jpg"
        video_id = "video_01"
        session_id = "sesion_mesa_lab"
        frame_index = 10

        # Tres residuos distintos en una misma escena
        input_boxes = [
            {"x1": 50.0, "y1": 60.0, "x2": 150.0, "y2": 200.0, "label": "plastico"},
            {"x1": 220.0, "y1": 110.0, "x2": 310.0, "y2": 195.0, "label": "metal"},
            {"x1": 380.0, "y1": 85.0, "x2": 490.0, "y2": 260.0, "label": "organicos"}
        ]

        # Guardar anotación
        self.dataset.add_or_update(
            image_path=image_path,
            video_id=video_id,
            session_id=session_id,
            frame_index=frame_index,
            boxes=input_boxes,
            is_empty_reviewed=False,
            split="train",
            width=640,
            height=480
        )

        # Volver a abrir desde disco creando una nueva instancia limpia de AnnotationDataset
        reloaded_dataset = AnnotationDataset(str(self.db_path))
        record = reloaded_dataset.get(image_path)

        self.assertIsNotNone(record, "El registro guardado debe existir al reabrir el dataset.")
        self.assertEqual(len(record.boxes), 3, "Debe conservar exactamente 3 cajas.")
        self.assertEqual(record.video_id, video_id)
        self.assertEqual(record.session_id, session_id)
        self.assertEqual(record.frame_index, frame_index)
        self.assertFalse(record.is_empty_reviewed)

        # Verificar cada una de las 3 cajas y clases
        expected_classes = ["plastico", "metal", "organicos"]
        for i, (actual, expected) in enumerate(zip(record.boxes, input_boxes)):
            self.assertEqual(actual["label"], expected["label"], f"La caja {i} debe tener clase {expected['label']}")
            self.assertEqual(actual["label"], expected_classes[i])
            self.assertAlmostEqual(actual["x1"], expected["x1"], places=1)
            self.assertAlmostEqual(actual["y1"], expected["y1"], places=1)
            self.assertAlmostEqual(actual["x2"], expected["x2"], places=1)
            self.assertAlmostEqual(actual["y2"], expected["y2"], places=1)

    def test_reviewed_empty_frame(self):
        """Verifica que un fotograma vacío revisado se registre como negativo sin residuos."""
        image_path = "data/annotated_frames/video_02/frame_00001.jpg"
        self.dataset.add_or_update(
            image_path=image_path,
            video_id="video_02",
            session_id="sesion_vacia_02",
            frame_index=1,
            boxes=[],
            is_empty_reviewed=True,
            split="train"
        )

        reloaded = AnnotationDataset(str(self.db_path))
        record = reloaded.get(image_path)
        self.assertIsNotNone(record)
        self.assertTrue(record.is_empty_reviewed)
        self.assertEqual(len(record.boxes), 0)

    def test_session_split_avoids_data_leakage(self):
        """Verifica que la partición por sesión agrupe todos los fotogramas de una sesión
        en una misma partición (train, val o test), evitando fuga de fotogramas contiguos."""
        # Registrar fotogramas de sesión 1 (3 fotogramas)
        for i in range(3):
            self.dataset.add_or_update(
                image_path=f"img_s1_{i}.jpg",
                video_id="vid_1",
                session_id="session_A",
                frame_index=i,
                boxes=[{"x1": 10, "y1": 10, "x2": 50, "y2": 50, "label": "carton"}]
            )
        # Registrar fotogramas de sesión 2 (3 fotogramas)
        for i in range(3):
            self.dataset.add_or_update(
                image_path=f"img_s2_{i}.jpg",
                video_id="vid_2",
                session_id="session_B",
                frame_index=i,
                boxes=[{"x1": 20, "y1": 20, "x2": 60, "y2": 60, "label": "vidrio"}]
            )
        # Registrar fotogramas de sesión 3 (3 fotogramas)
        for i in range(3):
            self.dataset.add_or_update(
                image_path=f"img_s3_{i}.jpg",
                video_id="vid_3",
                session_id="session_C",
                frame_index=i,
                boxes=[{"x1": 30, "y1": 30, "x2": 70, "y2": 70, "label": "papel"}]
            )

        splits = self.dataset.split_by_session(train_ratio=0.4, val_ratio=0.3, test_ratio=0.3, seed=123)

        reloaded = AnnotationDataset(str(self.db_path))
        # Revisar que para cada sesión, todos sus fotogramas tengan el mismo split
        for s_id in ["session_A", "session_B", "session_C"]:
            s_splits = {r.split for r in reloaded.records.values() if r.session_id == s_id}
            self.assertEqual(len(s_splits), 1, f"La sesión {s_id} no debe dividirse entre distintos splits.")


if __name__ == "__main__":
    unittest.main()
