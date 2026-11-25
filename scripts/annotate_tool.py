"""Herramienta de anotación de residuos para video e imágenes.

Permite:
- Abrir un video o secuencia de imágenes.
- Extraer y recorrer fotogramas con navegación interactiva.
- Dibujar cajas delimitadoras con el mouse (clic y arrastre).
- Asignar una de las 6 clases del proyecto mediante atajos (1 a 6).
- Corregir o eliminar recuadros (tecla 'd' para último, 'c' para limpiar).
- Marcar fotogramas vacíos revisados como ejemplos negativos sin residuos (tecla 'e').
- Guardar y volver a abrir anotaciones conservando coordenadas, clases y sesión de origen.

Uso:
    python -m scripts.annotate_tool --video ruta/al/video.mp4 --session sesion_01
    python -m scripts.annotate_tool --images_dir ruta/a/fotos/ --session sesion_01
"""
import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Configurar para evitar problemas con libiomp5md en Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.annotation_storage import AnnotationDataset, CLASSES, CLASS_DISPLAY

# Colores BGR por clase para visualización
COLORS = {
    "plastico": (255, 128, 0),     # Azul cielo / celeste
    "papel": (255, 255, 255),       # Blanco
    "carton": (42, 110, 180),       # Marrón / naranja oscuro
    "vidrio": (220, 220, 0),        # Cyan / turquesa
    "metal": (160, 160, 160),       # Gris metálico
    "organicos": (34, 139, 34)      # Verde bosque
}


class VideoAnnotator:
    def __init__(
        self,
        source: str,
        session_id: str,
        output_db: str = "data/annotations/annotations.json",
        frames_dir: str = "data/annotated_frames",
        fps_extract: int = 2
    ):
        self.source = Path(source)
        self.session_id = session_id
        self.video_id = self.source.stem
        self.db = AnnotationDataset(output_db)
        self.frames_dir = Path(frames_dir) / self.video_id
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps_extract = fps_extract

        self.frame_paths = []
        self.current_idx = 0
        self.current_class_idx = 0  # 0: plastico
        self.current_boxes = []
        self.is_empty_reviewed = False

        # Estado del ratón
        self.drawing = False
        self.start_point = None
        self.end_point = None

        self._load_or_extract_frames()

    def _load_or_extract_frames(self):
        if self.source.is_dir():
            # Cargar imágenes existentes
            exts = {".jpg", ".jpeg", ".png", ".webp"}
            self.frame_paths = sorted([p for p in self.source.iterdir() if p.suffix.lower() in exts])
            print(f"[Anotador] Se encontraron {len(self.frame_paths)} imágenes en {self.source}")
        elif self.source.is_file():
            # Es un archivo de video: extraer fotogramas si no existen
            existing = sorted(list(self.frames_dir.glob("frame_*.jpg")))
            if existing:
                self.frame_paths = existing
                print(f"[Anotador] Cargados {len(self.frame_paths)} fotogramas extraídos previamente.")
            else:
                print(f"[Anotador] Extrayendo fotogramas de {self.source} a ~{self.fps_extract} FPS...")
                cap = cv2.VideoCapture(str(self.source))
                if not cap.isOpened():
                    raise RuntimeError(f"No se pudo abrir el video {self.source}")
                video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                step = max(1, int(round(video_fps / self.fps_extract)))
                frame_count = 0
                extracted = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_count % step == 0:
                        out_name = self.frames_dir / f"frame_{extracted:05d}.jpg"
                        cv2.imwrite(str(out_name), frame)
                        self.frame_paths.append(out_name)
                        extracted += 1
                    frame_count += 1
                cap.release()
                print(f"[Anotador] Extracción completa: {len(self.frame_paths)} fotogramas guardados.")
        else:
            raise FileNotFoundError(f"Fuente no encontrada: {self.source}")

        if not self.frame_paths:
            raise RuntimeError("No hay fotogramas para anotar.")

    def _load_current_annotations(self):
        cur_path = str(self.frame_paths[self.current_idx].resolve())
        record = self.db.get(cur_path)
        if record:
            self.current_boxes = [dict(b) for b in record.boxes]
            self.is_empty_reviewed = record.is_empty_reviewed
        else:
            self.current_boxes = []
            self.is_empty_reviewed = False

    def save_current(self):
        cur_path = str(self.frame_paths[self.current_idx].resolve())
        img = cv2.imread(str(self.frame_paths[self.current_idx]))
        h, w = img.shape[:2] if img is not None else (480, 640)
        self.db.add_or_update(
            image_path=cur_path,
            video_id=self.video_id,
            session_id=self.session_id,
            frame_index=self.current_idx,
            boxes=self.current_boxes,
            is_empty_reviewed=self.is_empty_reviewed,
            width=w,
            height=h
        )

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                self.end_point = (x, y)
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                # Filtrar clics mínimos accidentales (< 8 píxeles)
                if (x2 - x1) >= 8 and (y2 - y1) >= 8:
                    current_label = CLASSES[self.current_class_idx]
                    self.current_boxes.append({
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "label": current_label
                    })
                    self.is_empty_reviewed = False
                    self.save_current()

    def run(self):
        window_name = "Herramienta de Anotacion - Vision-IA UCC Pasto"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        self._load_current_annotations()

        print("\n" + "=" * 60)
        print("HERRAMIENTA DE ANOTACIÓN INICIADA")
        print("Atajos de teclado:")
        print("  1-6    : Seleccionar clase (1:plástico, 2:papel, 3:cartón, 4:vidrio, 5:metal, 6:orgánicos)")
        print("  Ratón  : Clic y arrastrar para trazar recuadro del residuo")
        print("  'd'    : Eliminar último recuadro dibujado")
        print("  'c'    : Limpiar todos los recuadros del fotograma actual")
        print("  'e'    : Marcar/desmarcar fotograma como 'Revisado Vacío' (sin residuos)")
        print("  Espacio / 'n': Siguiente fotograma (guarda automáticamente)")
        print("  'p'    : Fotograma anterior (guarda automáticamente)")
        print("  's'    : Guardar anotaciones")
        print("  'q' / ESC: Guardar y salir")
        print("=" * 60 + "\n")

        while True:
            raw_img = cv2.imread(str(self.frame_paths[self.current_idx]))
            if raw_img is None:
                break
            display_img = raw_img.copy()

            # Dibujar cajas ya existentes
            for b in self.current_boxes:
                x1, y1, x2, y2 = int(b["x1"]), int(b["y1"]), int(b["x2"]), int(b["y2"])
                cls = b["label"]
                color = COLORS.get(cls, (0, 255, 0))
                cv2.rectangle(display_img, (x1, y1), (x2, y2), color, 2)
                tag = f"{CLASS_DISPLAY.get(cls, cls)}"
                cv2.putText(display_img, tag, (x1, max(20, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            # Dibujar caja en proceso de dibujo
            if self.drawing and self.start_point and self.end_point:
                cls = CLASSES[self.current_class_idx]
                color = COLORS.get(cls, (0, 255, 0))
                cv2.rectangle(display_img, self.start_point, self.end_point, color, 1)

            # Barra superior de estado e instrucciones
            h, w = display_img.shape[:2]
            header = np.zeros((48, w, 3), dtype=np.uint8)
            cur_cls = CLASSES[self.current_class_idx]
            cls_color = COLORS.get(cur_cls, (255, 255, 255))
            status_text = (
                f"[{self.current_idx + 1}/{len(self.frame_paths)}] "
                f"Video: {self.video_id} | Clase actual: [{self.current_class_idx + 1}] {CLASS_DISPLAY[cur_cls]} | "
                f"Cajas: {len(self.current_boxes)}"
            )
            if self.is_empty_reviewed:
                status_text += " [REVISADO VACIO]"

            cv2.putText(header, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.circle(header, (w - 30, 24), 10, cls_color, -1)

            combined = np.vstack([header, display_img])
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(20) & 0xFF

            # Teclas 1 a 6 para selección de clase
            if ord('1') <= key <= ord('6'):
                self.current_class_idx = key - ord('1')
                print(f"Clase activa: {CLASSES[self.current_class_idx]} ({CLASS_DISPLAY[CLASSES[self.current_class_idx]]})")
            elif key in (ord('n'), 32):  # 'n' o Espacio: siguiente
                self.save_current()
                if self.current_idx < len(self.frame_paths) - 1:
                    self.current_idx += 1
                    self._load_current_annotations()
            elif key in (ord('p'), 8):  # 'p' o Backspace: anterior
                self.save_current()
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self._load_current_annotations()
            elif key == ord('d'):  # Eliminar última caja
                if self.current_boxes:
                    removed = self.current_boxes.pop()
                    print(f"Caja eliminada: {removed['label']}")
                    self.save_current()
            elif key == ord('c'):  # Limpiar cajas del fotograma
                self.current_boxes = []
                self.save_current()
                print("Todas las cajas del fotograma eliminadas.")
            elif key == ord('e'):  # Alternar revisado vacío
                self.is_empty_reviewed = not self.is_empty_reviewed
                if self.is_empty_reviewed:
                    self.current_boxes = []
                self.save_current()
                print(f"Estado revisado vacío: {self.is_empty_reviewed}")
            elif key == ord('s'):
                self.save_current()
                print("Anotaciones guardadas correctamente.")
            elif key in (ord('q'), 27):  # 'q' o ESC: Salir
                self.save_current()
                print(f"Anotaciones guardadas en {self.db.file_path}. Saliendo...")
                break

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Herramienta de anotación interactiva de residuos")
    parser.add_argument("--source", type=str, default="", help="Ruta al archivo de video (.mp4) o carpeta de imágenes")
    parser.add_argument("--session", type=str, default="sesion_01", help="ID de la sesión de captura")
    parser.add_argument("--db", type=str, default="data/annotations/annotations.json", help="Ruta al archivo JSON de anotaciones")
    parser.add_argument("--fps", type=int, default=2, help="Fotogramas por segundo a extraer del video")
    args = parser.parse_args()

    if not args.source:
        print("Uso: python -m scripts.annotate_tool --source <video.mp4|carpeta_fotos> [--session sesion_01]")
        return

    annotator = VideoAnnotator(
        source=args.source,
        session_id=args.session,
        output_db=args.db,
        fps_extract=args.fps
    )
    annotator.run()


if __name__ == "__main__":
    main()
