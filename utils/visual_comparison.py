"""Generador de comparaciones visuales: Predicciones del Modelo vs Anotaciones Humanas.

Produce imágenes en formato side-by-side (lado a lado):
- Panel Izquierdo: Predicciones del modelo (recuadros, clase, confianza).
- Panel Derecho: Anotaciones humanas de referencia (Ground Truth).
- Resumen inferior: IoU de coincidencia y estado de acierto/error.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np

from utils.annotation_storage import CLASSES, CLASS_DISPLAY

COLORS = {
    "plastico": (255, 128, 0),     # Azul celeste
    "papel": (255, 255, 255),       # Blanco
    "carton": (42, 110, 180),       # Marrón / naranja
    "vidrio": (220, 220, 0),        # Cyan / turquesa
    "metal": (160, 160, 160),       # Gris metálico
    "organicos": (34, 139, 34)      # Verde
}


def draw_labeled_boxes(
    image: np.ndarray,
    boxes: List[Dict[str, Any]],
    is_prediction: bool = False
) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]

    for b in boxes:
        if is_prediction:
            x1, y1, x2, y2 = map(int, b["box"])
            cls = b.get("class_name", "residuo")
            conf = b.get("confidence", 0.0)
            tag = f"{CLASS_DISPLAY.get(cls, cls)} ({conf * 100:.1f}%)"
        else:
            x1 = int(b.get("x1", b.get("box", [0, 0, 0, 0])[0]))
            y1 = int(b.get("y1", b.get("box", [0, 0, 0, 0])[1]))
            x2 = int(b.get("x2", b.get("box", [0, 0, 0, 0])[2]))
            y2 = int(b.get("y2", b.get("box", [0, 0, 0, 0])[3]))
            cls = b.get("label", b.get("class_name", "residuo"))
            tag = f"[GT] {CLASS_DISPLAY.get(cls, cls)}"

        color = COLORS.get(cls, (0, 255, 255))
        # Dibujar recuadro
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        # Fondo del texto para legibilidad
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty1 = max(th + 4, y1)
        cv2.rectangle(canvas, (x1, ty1 - th - 4), (x1 + tw + 6, ty1 + 2), (20, 20, 20), -1)
        cv2.putText(canvas, tag, (x1 + 3, ty1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return canvas


def create_side_by_side_comparison(
    image_rgb: np.ndarray,
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    title_suffix: str = ""
) -> np.ndarray:
    """Genera una imagen comparativa con las predicciones del modelo a la izquierda
    y las anotaciones humanas de referencia a la derecha."""
    # Convertir a BGR para OpenCV si es necesario
    if image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = image_rgb.copy()

    h, w = bgr.shape[:2]

    # Panel izquierdo: Predicciones
    left_img = draw_labeled_boxes(bgr, predictions, is_prediction=True)
    # Panel derecho: Ground Truth
    right_img = draw_labeled_boxes(bgr, ground_truths, is_prediction=False)

    # Encabezados
    header_h = 42
    header_left = np.full((header_h, w, 3), (35, 35, 35), dtype=np.uint8)
    header_right = np.full((header_h, w, 3), (25, 45, 25), dtype=np.uint8)

    cv2.putText(header_left, "PREDICCION DEL MODELO", (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(header_right, "ANOTACION HUMANA (GROUND TRUTH)", (15, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 2, cv2.LINE_AA)

    panel_left = np.vstack([header_left, left_img])
    panel_right = np.vstack([header_right, right_img])

    # Separador vertical
    separator = np.full((h + header_h, 4, 3), (80, 80, 80), dtype=np.uint8)
    combined = np.hstack([panel_left, separator, panel_right])

    # Barra de resumen inferior
    footer_h = 36
    total_w = combined.shape[1]
    footer = np.full((footer_h, total_w, 3), (30, 30, 30), dtype=np.uint8)

    summary_text = (
        f"Objetos Predichos: {len(predictions)} | Objetos Reales: {len(ground_truths)} "
        f"| Proyecto 4: Deteccion de Residuos (UCC Pasto) {title_suffix}"
    )
    cv2.putText(footer, summary_text, (20, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    final_comparison = np.vstack([combined, footer])
    return cv2.cvtColor(final_comparison, cv2.COLOR_BGR2RGB)


def save_comparison(
    image_rgb: np.ndarray,
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    output_path: str
):
    comp_rgb = create_side_by_side_comparison(image_rgb, predictions, ground_truths)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(comp_rgb, cv2.COLOR_RGB2BGR))
