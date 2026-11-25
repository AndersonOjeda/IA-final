"""Calibración empírica conjunta de umbrales (Confianza y NMS) para maximizar F1-Score.

Realiza un barrido 2D sobre el conjunto de validación:
- Umbral de confianza theta_conf en [0.10, 0.60]
- Umbral de NMS theta_nms en [0.30, 0.60]
- Encuentra la combinación óptima (theta_conf*, theta_nms*) bajo coincidencia IoU >= 0.5.
- Guarda la configuración óptima para la aplicación y la inferencia.

Uso:
    python -m scripts.calibrate_threshold [--checkpoint runs/waste_detector/best_model.pth]
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from utils.annotation_storage import AnnotationDataset, CLASSES
from training.multi_waste_detector import WasteDetectorNet, decode_detections, DetectorEvaluator
from scripts.train_waste_detector import WasteDataset, collate_fn


def calibrate_optimal_thresholds(
    checkpoint_path: str = "runs/waste_detector/best_model.pth",
    db_path: str = "data/annotations/annotations.json"
):
    path = Path(checkpoint_path)
    if not path.exists():
        print(f"❌ Checkpoint no encontrado en {path}. Entrena primero el detector con:")
        print("   python -m scripts.train_waste_detector")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("state_dict") or payload.get("model_state_dict") or payload
    model = WasteDetectorNet(num_classes=payload.get("num_classes", len(CLASSES))).to(device)
    model.load_state_dict(state)
    model.eval()

    if not Path(db_path).exists() and Path("data/benchmark/annotations.json").exists():
        db_path = "data/benchmark/annotations.json"

    dataset = AnnotationDataset(db_path)
    records = [r for r in dataset.records.values() if r.split in ("val", "train")]
    if not records:
        print("[Calibrador] Usando datos de prueba técnica...")
        from utils.annotation_storage import AnnotationRecord
        records = [
            AnnotationRecord(
                image_path=f"syn_val_{i}.jpg",
                video_id="val_vid",
                session_id="val_session",
                frame_index=i,
                boxes=[
                    {"x1": 30, "y1": 40, "x2": 110, "y2": 160, "label": "plastico"},
                    {"x1": 140, "y1": 80, "x2": 210, "y2": 180, "label": "metal"}
                ] if i % 3 != 0 else [],
                width=256,
                height=256
            )
            for i in range(15)
        ]

    val_ds = WasteDataset(records, is_train=False)

    conf_thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    nms_thresholds = [0.35, 0.45, 0.55]

    print("=" * 76)
    print("CALIBRACIÓN BIDIMENSIONAL CON RESTRICCIÓN DE ALTO RECALL (IoU >= 0.5)")
    print("=" * 76)
    print(f"{'Conf (theta_c)':>15} {'NMS (theta_nms)':>16} {'Precisión':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 76)

    best_conf = 0.15
    best_nms = 0.45
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    sweep_results = []

    # Candidato con restricción Recall >= 90%
    high_recall_candidate = None
    max_recall_seen = 0.0
    best_recall_candidate = None

    # Extraer salidas una sola vez
    cached_data = []
    with torch.no_grad():
        for idx in range(len(val_ds)):
            tensor, _, _, _, _, raw_boxes, _ = val_ds[idx]
            batch_tensor = tensor.unsqueeze(0).to(device)
            raw_out = model(batch_tensor)
            cached_data.append((raw_out, raw_boxes))

    evaluator = DetectorEvaluator(iou_threshold=0.5)

    for nms_t in nms_thresholds:
        for conf_t in conf_thresholds:
            val_gts = []
            val_preds = []
            for raw_out, raw_boxes in cached_data:
                preds = decode_detections(
                    raw_out, conf_threshold=conf_t, nms_iou_threshold=nms_t, img_size=256, class_aware_nms=True
                )[0]
                val_gts.append(raw_boxes)
                val_preds.append(preds)

            metrics = evaluator.evaluate(val_gts, val_preds)
            p = metrics["mean_precision"]
            r = metrics["mean_recall"]
            f1 = metrics["mean_f1"]

            sweep_results.append({
                "conf_threshold": conf_t, "nms_threshold": nms_t,
                "precision": p, "recall": r, "f1": f1
            })

            print(f"{conf_t:>15.2f} {nms_t:>16.2f} {p * 100:>11.1f}% {r * 100:>11.1f}% {f1 * 100:>11.1f}%")

            if r > max_recall_seen:
                max_recall_seen = r
                best_recall_candidate = {"conf": conf_t, "nms": nms_t, "p": p, "r": r, "f1": f1}

            # Restricción 1: Si cumple Recall >= 0.90, seleccionar el de mayor F1/Precision
            if r >= 0.90:
                if high_recall_candidate is None or f1 > high_recall_candidate["f1"]:
                    high_recall_candidate = {"conf": conf_t, "nms": nms_t, "p": p, "r": r, "f1": f1}

            # Criterio F1 balanceado general
            if f1 > best_f1 or (f1 == best_f1 and p > best_p):
                best_f1 = f1
                best_conf = conf_t
                best_nms = nms_t
                best_p = p
                best_r = r

    # Si se encontró candidato con Recall >= 90%, se prioriza
    if high_recall_candidate is not None:
        selected_conf = high_recall_candidate["conf"]
        selected_nms = high_recall_candidate["nms"]
        selected_p = high_recall_candidate["p"]
        selected_r = high_recall_candidate["r"]
        selected_f1 = high_recall_candidate["f1"]
        mode_note = "Restricción Recall >= 90% alcanzada"
    else:
        # Si ningún umbral alcanza 90% debido a la resolución espacial y tamaño del dataset,
        # seleccionar el umbral de máxima sensibilidad/recall posible
        selected_conf = best_conf
        selected_nms = best_nms
        selected_p = best_p
        selected_r = best_r
        selected_f1 = best_f1
        mode_note = f"Recall máximo alcanzado en val: {max_recall_seen * 100:.1f}%. Se calibra para máximo F1."

    print("-" * 76)
    print(f"⭐ UMBRALES ÓPTIMOS CALIBRADOS ({mode_note}):")
    print(f"   • Confianza (theta_conf*): {selected_conf:.2f}")
    print(f"   • NMS IoU (theta_nms*)  : {selected_nms:.2f}")
    print(f"   -> F1-Score: {selected_f1 * 100:.1f}% | Precisión: {selected_p * 100:.1f}% | Recall: {selected_r * 100:.1f}%")
    print("=" * 76)

    out_file = path.parent / "calibrated_threshold.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_conf_threshold": selected_conf,
            "optimal_nms_threshold": selected_nms,
            "best_f1": selected_f1,
            "best_precision": selected_p,
            "best_recall": selected_r,
            "high_recall_candidate": high_recall_candidate,
            "max_recall_seen": max_recall_seen,
            "best_recall_candidate": best_recall_candidate,
            "sweep_results": sweep_results
        }, f, indent=2)
    print(f"Configuración guardada en: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Calibrar umbrales óptimos de detección")
    parser.add_argument("--checkpoint", type=str, default="runs/waste_detector/best_model.pth")
    parser.add_argument("--db", type=str, default="data/annotations/annotations.json")
    args = parser.parse_args()

    calibrate_optimal_thresholds(args.checkpoint, args.db)


if __name__ == "__main__":
    main()
